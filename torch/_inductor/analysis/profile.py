import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

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


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
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

    report()


def diff_profiles(diff_path1: str, diff_path2: str) -> None:
    from collections import defaultdict

    def parse(data: dict[str, Any]) -> KernelNameMap:
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
        return name_map

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

    def parse_helper(filename: str) -> defaultdict[str, OrderedSet[KernelStats]]:
        with open(filename) as f:
            data = json.load(f)
        return parse(data)

    other_nm = parse_helper(diff_path1)
    diff_nm = parse_helper(diff_path2)
    combine_name_maps(diff_path1, other_nm, diff_path2, diff_nm)


class ParseException(RuntimeError):
    pass


def augment_trace_with_inductor_meta(
    input_file_path: str, output_file_path: str
) -> None:
    """
    Many of the important ops don't go through the triton flops calculation, because they're external kernels. Instead, we will
    augment the profile after it runs using the input information
    """
    # Load the JSON file
    with open(input_file_path) as f:
        data = json.load(f)

    ATEN_PREFIX = "aten::"

    def calculate_flops(event: dict[str, Any]) -> int:
        op_name = name[len(ATEN_PREFIX) :]
        flop_function = flop_registry[getattr(torch.ops.aten, op_name)]
        input_sizes = filter(lambda x: x != [], event["args"]["Input Dims"])
        return flop_function(*input_sizes)

    def estimate_bandwidth(event: dict[str, Any]) -> float:
        """
        This estimate isn't the best because it doesn't know if two input buffers are the same buffer, leading to an
        overestimate of the real achieved bandwidth.
        """
        sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input Type"])
        bw = 0
        for size, tipe in sizes_and_types:
            dtype = getattr(torch, tipe)
            bw += dtype.itemsize * math.prod(size)
        return bw

    for event in data["traceEvents"]:
        if "name" not in event:
            raise ParseException("no name element in event")
        name = event["name"]
        if name.startswith(ATEN_PREFIX):
            event["args"]["kernel_flops"] = calculate_flops(event)
            event["args"]["kernel_bandwidth_estimate"] = estimate_bandwidth(event)

    with open(output_file_path, "w") as f:
        json.dump(data, f)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diff",
        "-d",
        type=str,
        nargs=2,
        help="Two json traces to compare with",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        type=str,
        nargs=2,
        metavar=("input_file", "output_file"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )

    args = parser.parse_args()

    if args.diff:
        diff_profiles(args.diff[0], args.diff[1])

    if args.augment:
        augment_trace_with_inductor_meta(args.augment[0], args.augment[1])


if __name__ == "__main__":
    main()
