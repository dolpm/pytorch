import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, Optional, Sequence

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry
from tabulate import tabulate




PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"
ATEN_PREFIX = "aten::"


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float



# adapters convert the json trace into a format that works with flops_counter
adapters_map: dict[str, Any] = {}

def register_adapter(aten: Union[str, list[str]]):
    def decorator(func):
        global _adapters_map
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        if isinstance(aten, str):
            adapters_map[aten] = wrapper
        else:
            for at in aten:
                adapters_map[at] = wrapper
        return wrapper
    return decorator

@register_adapter(["convolution", "_convolution"])
def conv_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    tmp[6] = bool(tmp[6])
    return tuple(tmp), {}


def default_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}

@register_adapter("addmm")
def addmm_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}

@register_adapter("bmm")
def bmm_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    return tuple(tmp[:2]), {}

@register_adapter("baddbmm")
def baddbmm_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}

@register_adapter("mm")
def mm_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


def _calculate_flops(event: dict[str, Any]) -> int:
    op_name = event["name"][len(ATEN_PREFIX):]
    op_obj = getattr(torch.ops.aten, op_name)
    if not op_obj in flop_registry:
        return 0

    flop_function = flop_registry[op_obj]
    
    input_shapes = event["args"]["Input Dims"]
    concrete = event["args"]["Concrete Inputs"]
    if op_name in adapters_map:
        args, kwargs = adapters_map[op_name](input_shapes, concrete)
    else:
        breakpoint()
        args, kwargs = default_adapter(input_shapes, concrete)
    return flop_function(*args, **kwargs)
def _estimate_bandwidth(event: dict[str, Any]) -> float:
    """
    This estimate isn't the best because it doesn't know if two input buffers are the same buffer, leading to an
    overestimate of the real achieved bandwidth.
    """
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, tipe in sizes_and_types:
        if not hasattr(torch, tipe):
            isize = 0
        else:
            isize = getattr(torch, tipe).itemsize
        try:
            bw += isize * math.prod(flatten(size))
        except:
            breakpoint()
    return bw

def _augment_trace_helper(data: dict[str, Any], nruns: int) -> dict[str, Any]:
    for event in data["traceEvents"]:
        if "name" not in event:
            raise ParseException("no name element in event")
        name = event["name"]
        if name.startswith(ATEN_PREFIX):
            event["args"]["kernel_flops"] = _calculate_flops(event)
            event["args"]["kernel_bandwidth_estimate"] = _estimate_bandwidth(event)
    return data

@dataclass(frozen=True)
class DeviceInfo:
    flops: dict[torch.dtype, int]
    dram_bw: float
    

_device_mapping: dict[str, DeviceInfo] = {
    "Nvidia H100": DeviceInfo({}, 10)
}
def lookup_device_info(name: str) -> "DeviceInfo":
    """
    problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device.
    """
    if name not in _device_mapping:
        raise RuntimeError(f"Unsupported device in profile: {name}, if it's a more obscure device, consider contributing to _device_mapping.")
    return _device_mapping[name]

_dtype_map = {
    "float": torch.float,
    "int": torch.int,
    "long": torch.long,
    "long int": torch.long,
}

@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float
    achieved_flops: float
    achieved_bandwidth: float
KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]
@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: DeviceInfo
    stats: KernelNameMap
DeviceMap = dict[int, Device]
Table = tabulate
class JsonProfile:
    """operations on json perfetto traces"""
    _devices: DeviceMap
    def __init__(self, path: str, nruns: int, device_name: Optional[str] = None, benchmark_name: Optional[str] = None):
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.nruns = nruns
        self.device_name = device_name
        self.benchmark_name = benchmark_name
        self._create_devices()

    def convert_dtype(self, input_sizes: list[str], input_types: list[str], concrete_inputs: list[str]) -> torch.dtype:
        """
        Each op has a list of dtypes for each input arg. We need to convert these into a single dtype for flop estimation.
        Issues:
         - converting the strings to concrete torch.dtypes
         - What if we have float32, float, float16 all in the inputs? Our choice is to use the largest buffer dtype.
        """
        assert len(input_sizes) == len(input_types)
        assert len(input_types) == len(concrete_inputs)
        if len(input_sizes) == 0:
            raise RuntimeError("Empty input_sizes and input_types")

        def parse_list(lst: str) -> list[int]:
            lst = lst.replace('[', '').replace(']', '')
            substrings = lst.split(',')
            return [int(substring.strip()) for substring in substrings]

        biggest_size = 0
        biggest_index = 0
        for i in range(len(input_sizes)):
            if concrete_inputs[i] != "":
                # concrete inputs are usually small tensors, so we can just skip
                continue
            my_size = input_sizes[i]
            total_size = sum(parse_list(my_size))
            if total_size > biggest_size:
                biggest_size = total_size
                biggest_index = i
        ret_type = input_types[biggest_index]
        if ret_type in _dtype_map:
            return _dtype_map[ret_type]
        raise RuntimeError(f"Unknown type: {ret_type}. Please add to _dtype_map.")

    def _create_devices(self):
        self._devices = {dev["id"]: Device(dev["name"], dev["id"], lookup_device_info(dev["name"]), defaultdict(OrderedSet)) for dev in self.data["deviceProperties"]}

    def calculate_flops(self, event: dict[str, Any]) -> int:
        return _calculate_flops(event)

    def estimate_bandwidth(self, event: dict[str, Any]) -> float:
        """
        This estimate isn't the best because it doesn't know if two input buffers are the same buffer, leading to an
        overestimate of the real achieved bandwidth.
        """
        return _estimate_bandwidth(event)

    def augment_trace(self) -> None:
        self.data = _augment_trace_helper(self.data, self.nruns)

    def _compute_stats(self) -> None:
        """populates the name -> stats map"""
        for event in self.events:
            if (
                "args" in event
                and event["cat"] == "kernel"
                and "kernel_flops" in event["args"]
                and "kernel_bandwidth" in event["args"]
            ):
                dev = self._devices[event["args"]["device"]]
                latency = event["dur"]
                op_bw = event["args"]["kernel_bandwidth"]
                op_flops = event["args"]["kernel_flops"]
                dtype = self.convert_dtype(event["args"]["Input Dims"], event["args"]["Input type"], event["args"]["Concrete Inputs"])
                
                # TODO check units here
                # TODO this formula is wrong
                achieved_bandwidth =  op_bw * latency / dev.info.dram_bw
                achieved_flops = op_flops * latency / dev.info.flops[dtype]
                dev.stats[event["name"]].add(KernelStats(op_flops, op_bw, latency, achieved_bandwidth, achieved_flops))
    def _create_single_table(self, dev: Device) -> Table:
        """create a table with the devices mapped to indicies"""
        pass

    def _create_tables(self, devs: DeviceMap) -> dict[int, Table]:
        return {idx: self._create_single_table(dev) for idx, dev in devs.items()} 
    def _combine_tables(self, table1: Table, table2: Table) -> Table:
        pass

    def report(self, other: Optional["JsonProfile"] = None) -> str:
        if other is not None:
            self._compute_stats()
            other._compute_stats()

            self_tables = self._create_tables(self._devices)
            other_tables = self._create_tables(other._devices)
            indicies1 = OrderedSet(self._devices.keys())
            indicies2 = OrderedSet(other._devices.keys())
            combined_tables: dict[int, Table] = {}
            for comb_index in indicies1 | indicies2:
                combined_tables[comb_index] = self._combine_tables(self_tables[comb_index], other_tables[comb_index])
            for comb_index in indicies1 - indicies2:
                combined_tables[comb_index] = self_tables[comb_index]
            for comb_index in indicies2 - indicies1:
                combined_tables[comb_index] = other_tables[comb_index]

            ret = []
            for idx, table in combined_tables.items():
                ret.append(f"Device {idx}:\n{table}")
            return "\n".join(ret)
        self._compute_stats()

        self_tables = self._create_tables(self._devices)

        ret = []
        for idx, table in self_tables.items():
            ret.append(f"Device {idx}:\n{table}")
        return "\n".join(ret)
        #print(tabulate(table, headers=headers, tablefmt="grid"))

    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(self.data, f)



def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList | dict[str, Any],
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)), (
            f"{list(all_events.keys())}"
        )

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    #breakpoint()
    report()


class ParseException(RuntimeError):
    pass


def flatten(lst: Sequence[Union[int, Sequence[int]]]) -> Sequence[int]:
    """Flatten a nested list of integers."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def _augment_trace_with_inductor_meta(
    input_file_path: str, output_file_path: str, nruns: int
) -> None:
    """
    Many of the important ops don't go through the triton flops calculation, because they're external kernels. Instead, we will
    augment the profile after it runs using the input information
    """
    p = JsonProfile(input_file_path, nruns)
    # process
    processed_data = p.augment_trace()
    # store
    p.dump(output_file_path)




def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=4,
        help="Two json traces to compare with, specified as <file1> <nruns1> <file2> <nruns2>",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        type=str,
        nargs=2,
        metavar=("input_file", "output_file"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=2,
        help="Run analysis on a single trace, specified as <file> <nruns>",
    )
    args = parser.parse_args()

    if args.diff:
        # todo add name to diff
        p1 = JsonProfile(args.diff[0], int(args.diff[1]))
        p1.augment_trace()
        p2 = JsonProfile(args.diff[2], int(args.diff[3]))
        p2.augment_trace()
        print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(args.analysis[0], args.analysis[1])
        p1.augment_trace()
        print(p1.report())
    if args.augment_trace:
        _augment_trace_with_inductor_meta(args.augment[0], args.nruns, args.augment[1])


if __name__ == "__main__":
    main()
