import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, Optional

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float


PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]

# adapters convert the json trace into a format that works with flops_counter
adapters_map: dict[str, Any] = {}

def register_adapter(aten: str | list[str]):
    def decorator(func):
        global _adapters_map
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        if isinstance(aten, str):
            _adapters_map[aten] = wrapper
        else:
            for at in aten:
                _adapters_map[at] = wrapper
        return wrapper
    return decorator

@register_adapter(["convolution", "_convolution"])
def conv_adapter(shapes: tuple[Any], concrete: tuple[Any]) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    tmp[6] = bool(concrete[6])
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

def _augment_trace_helper(data: dict[str, Any], nruns: int) -> dict[str, Any]:
    ATEN_PREFIX = "aten::"
    get_kernels(data) 

    for event in data["traceEvents"]:
        if "name" not in event:
            raise ParseException("no name element in event")
        name = event["name"]
        if name.startswith(ATEN_PREFIX):
            event["args"]["kernel_flops"] = calculate_flops(event)
            event["args"]["kernel_bandwidth_estimate"] = estimate_bandwidth(event)
    return data

def _calculate_flops(event: dict[str, Any]) -> int:
    op_name = name[len(ATEN_PREFIX) :]
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
def estimate_bandwidth(event: dict[str, Any]) -> float:
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
class JsonProfile:
    """operations on json traces"""
    _stats: KernelNameMap
    def __init__(self, path: str, nruns: int, device_name: Optional[str] = None, benchmark_name: Optinal[str] = None):
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
        self.nruns = nruns
        self.device_name = device_name
        self.benchmark_name = benchmark_name

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
    def _compute_events(self) -> None:
        pass
    def _compute_stats(self) -> None:
        """populates the name -> stats map"""
        if self._stats is not None:
            name_map: KernelNameMap = defaultdict(OrderedSet)
            for event in data["traceEvents"]:
                if (
                    "args" in event
                    and "kernel_flops" in event["args"]
                    and "kernel_bandwidth" in event["args"]
                ):
                    name_map[event["name"]].add(
                        KernelStats(
                            event["args"]["kernel_flops"], event["args"]["kernel_bandwidth"]
                        )
                    )
            self._stats
    def report(self, other: Optional["JsonProfile"] = None) -> str:
        self._compute_stats()
        self._compute_events()
        def combine_name_maps(
            filename1: str,
            name_map1: KernelNameMap,
            filename2: str,
            name_map2: KernelNameMap,
        ) -> None:
            from tabulate import tabulate

            combined_table = {}

            # Get all unique names from both name maps
            all_names = OrderedSet(list(name_map1.keys()) + list(name_map2.keys()))

            for name in all_names:
                stats1 = name_map1.get(name, OrderedSet())
                stats2 = name_map2.get(name, OrderedSet())

                flops_avg1 = (
                    sum(stat.flops for stat in stats1) / len(stats1) if stats1 else 0
                )
                bw_avg1 = sum(stat.bw for stat in stats1) / len(stats1) if stats1 else 0

                flops_avg2 = (
                    sum(stat.flops for stat in stats2) / len(stats2) if stats2 else 0
                )
                bw_avg2 = sum(stat.bw for stat in stats2) / len(stats2) if stats2 else 0

                combined_table[name] = [flops_avg1, bw_avg1, flops_avg2, bw_avg2]

            headers = [
                "Kernel Name",
                f"{filename1} FLOPS",
                f"{filename1} Bandwidth",
                f"{filename2} FLOPS",
                f"{filename2} Bandwidth",
            ]
            table = [[name] + values for name, values in combined_table.items()]
            print(tabulate(table, headers=headers, tablefmt="grid"))
    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(processed_data, f)



def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList | dict[str, Any],
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    # breakpoint()
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

    #breakpoint()
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


def flatten(lst: list[Union[int, list[int]]]) -> list[int]:
    """Flatten a nested list of integers."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def get_kernels(data: dict[str, Any]) -> dict[str, list[ProfileEvent]]:
    #breakpoint()
    pass


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
