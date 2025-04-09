# Owner(s): ["module: inductor"]

import json
import tempfile
import uuid
from io import StringIO
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch.utils.flop_counter
from torch._inductor.analysis.profile import _augment_trace_helper, main
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


example_profile = """
{
  "schemaVersion": 1,
  "deviceProperties": [
    {
      "id": 0, "name": "NVIDIA H100", "totalGlobalMem": 101997215744,
      "computeMajor": 9, "computeMinor": 0,
      "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 2048,
      "regsPerBlock": 65536, "warpSize": 32,
      "sharedMemPerBlock": 49152, "numSms": 132
    , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 232448, "sharedMemPerMultiprocessor": 233472
    }
  ],
  "cupti_version": 24,
  "cuda_runtime_version": 12060,
  "with_flops": 1,
  "record_shapes": 1,
  "cuda_driver_version": 12040,
  "profile_memory": 1,
  "trace_id": "301995E163ED42048FBD783860E6E7DC",
  "displayTimeUnit": "ms",
  "baseTimeNanoseconds": 1743521598000000000,
  "traceEvents": [
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::convolution", "pid": 1147039, "tid": 1147039,
    "ts": 198093488368.463, "dur": 425.453,
    "args": {
      "External id": 1340,"Sequence number": 0, "Fwd thread id": 0, "Record function id": 0, "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", "False", "[0, 0]", "1"], "Input type": ["float", "float", "", "ScalarList", "ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar"], "Input Strides": [[150528, 1, 672, 3], [147, 1, 21, 3], [], [], [], [], [], [], []], "Input Dims": [[1, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], []], "Ev Idx": 1339
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_convolution", "pid": 1147039, "tid": 1147039,
    "ts": 198093488444.498, "dur": 341.867,
    "args": {
      "External id": 1341,"Record function id": 0, "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", "False", "[0, 0]", "1", "False", "False", "True", "True"], "Input type": ["float", "float", "", "ScalarList", "ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar", "Scalar", "Scalar", "Scalar", "Scalar"], "Input Strides": [[150528, 1, 672, 3], [147, 1, 21, 3], [], [], [], [], [], [], [], [], [], [], []], "Input Dims": [[1, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], [], [], [], [], []], "Ev Idx": 1340
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::addmm", "pid": 1147039, "tid": 1147039,
    "ts": 198093513655.849, "dur": 251.130,
    "args": {
      "External id": 1619,"Sequence number": 0, "Fwd thread id": 0, "Record function id": 0, "Concrete Inputs": ["", "", "", "1", "1", ""], "Input type": ["float", "float", "float", "Scalar", "Scalar", "float"], "Input Strides": [[1], [0, 1], [1, 2048], [], [], [1000, 1]], "Input Dims": [[1000], [1, 2048], [2048, 1000], [], [], [1, 1000]], "Ev Idx": 1618
    }
  }
  {
    # this one to test unknown aten
    "ph": "X", "cat": "cpu_op", "name": "aten::foo", "pid": 1147039, "tid": 1147039,
    "ts": 198093488444.498, "dur": 341.867,
    "args": {
      "External id": 1341,"Record function id": 0, "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", "False", "[0, 0]", "1", "False", "False", "True", "True"], "Input type": ["float", "float", "", "ScalarList", "ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar", "Scalar", "Scalar", "Scalar", "Scalar"], "Input Strides": [[150528, 1, 672, 3], [147, 1, 21, 3], [], [], [], [], [], [], [], [], [], [], []], "Input Dims": [[1, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], [], [], [], [], []], "Ev Idx": 1340
    }
  },
],
  "traceName": "/tmp/compiled_module_profile.json"
}
"""


def verify_flops(self, expected_flops, out_profile):
    for i in len(out_proflie["traceEvents"]):
        self.assertEqual(
            out_profile["traceEvents"][i]["args"]["kernel_flops"], expected_flops[i]
        )


def random_tensor(size, dtype, **kwargs):
    if dtype in [torch.half, torch.bfloat16, torch.float, torch.double]:
        return torch.randn(size, dtype=dtype, **kwargs)
    elif dtype in [torch.uint8, torch.int8, torch.short, torch.int, torch.long]:
        return torch.randint(0, 100, size, dtype=dtype, **kwargs)
    else:
        raise ValueError("Unsupported data type")


def cT(device, dtype):
    def T(*shape, requires_grad=False):
        return random_tensor(
            shape, requires_grad=requires_grad, device=device, dtype=dtype
        )

    return T


def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


TMP_DIR = tempfile.mkdtemp()


def trace_files():
    TRACE1 = f"{TMP_DIR}/trace1-{uuid.uuid4()}.txt"
    TRACE2 = f"{TMP_DIR}/trace2-{uuid.uuid4()}.txt"
    return TRACE1, TRACE2


def omni_model(device, dtype):
    T = cT(device, dtype)

    def model():
        # Create input tensors
        input_conv = T(1, 3, 28, 28)  # batch_size, channels, height, width
        conv_weight = T(
            6, 3, 5, 5
        )  # output_channels, input_channels, kernel_height, kernel_width

        mat1 = T(2, 3)  # Adjusted matrix 1 for matrix multiplication
        mat2 = T(3, 4)  # matrix 2 for matrix multiplication

        batch_mat1 = T(1, 3, 4)  # batched matrix 1 for batch matrix multiplication
        batch_mat2 = T(
            1, 4, 24 * 24
        )  # batched matrix 2 for batch matrix multiplication

        # Convolution operation
        conv_output = F.conv2d(input_conv, conv_weight)

        # a pointwise op
        conv_output = conv_output * 10

        # Matrix multiplication (addmm) operation
        addmm_output = torch.addmm(
            torch.zeros(2, 4, device=mat1.device, dtype=mat1.dtype), mat1, mat2
        )

        # Batch matrix multiplication (bmm) operation
        bmm_output = torch.bmm(batch_mat1, batch_mat2)

        # Batch addition matrix multiplication (baddbmm) operation
        baddbmm_output = torch.baddbmm(
            torch.zeros(1, 3, 576, device=batch_mat1.device, dtype=batch_mat1.dtype),
            batch_mat1,
            batch_mat2,
        )

        mm_output = torch.mm(mat1, mat2)

        return torch.cat(
            [
                conv_output.flatten(),
                addmm_output.flatten(),
                bmm_output.flatten(),
                baddbmm_output.flatten(),
                mm_output.flatten(),
            ]
        )

    return torch.compile(
        model, options={"benchmark_kernel": True, "profile_bandwidth": True}
    )


prefix = ["profile.py"]


class TestAnalysis(TestCase):
    def test_noop(self):
        with (
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            patch("sys.argv", [*prefix]) as mock_argv,
        ):
            main()
            self.assertEqual(mock_stdout.getvalue(), "")

    @dtypes(torch.float, torch.double)
    def test_diff(self, device, dtype):
        """
        diff, testing out the nruns feature too.
        """
        om = omni_model(device, dtype)
        REPEAT = 5
        trace1, trace2 = trace_files()
        with torch.profiler.profile(record_shapes=True) as p:
            om()
        p.export_chrome_trace(trace1)

        with torch.profiler.profile(record_shapes=True) as p:
            for _ in range(REPEAT):
                om()
        p.export_chrome_trace(trace2)

        breakpoint()
        # patch('sys.stdout', new_callable=StringIO) as mock_stdout,
        with patch(
            "sys.argv", [*prefix, "--diff", trace1, "1", trace2, str(REPEAT)]
        ) as mock_argv:
            main()
            # self.assertEqual(mock_stdout.getvalue(), "")

    def test_augment_trace_helper(self):
        js = json.loads(example_profile)
        out_profile = _augment_trace_helper(js, 1)
        expected_flops = [1, 2, 3]
        verify_flops(self, expected_flops, out_profile)

    @dtypes(torch.float, torch.double)
    def test_augment_trace_helper_args(self, device, dtype):
        om = omni_model(device, dtype)
        with torch.profiler.profile(record_shapes=True) as p:
            om()
        trace1, trace2 = trace_files()
        p.export_chrome_trace(trace1)
        # patch('sys.stdout', new_callable=StringIO) as mock_stdout,
        with patch(
            "sys.argv", [*prefix, "--augment_trace", trace1, trace2]
        ) as mock_argv:
            main()
            # self.assertEqual(mock_stdout.getvalue(), "")

    @dtypes(torch.float, torch.double)
    def test_augment_trace_against_flop_counter(self, device, dtype):
        om = omni_model(device, dtype)
        comp_omni = torch.compile(
            om, options={"benchmark_kernel": True, "profile_bandwidth": True}
        )

        with torch.profiler.profile(record_shapes=True) as p:
            comp_omni()

        with FlopCounterMode() as mode:
            comp_omni()
        PROFILE_DIR = tempfile.gettempdir()
        in_path = f"{PROFILE_DIR}/test_profile.json"
        out_path = f"{PROFILE_DIR}/out_profile.json"
        p.export_chrome_trace(in_path)
        augment_trace_with_inductor_meta(in_path, out_path, 1)

        with open(out_path) as f:
            out_profile = json.load(f)

        flop_counts = mode.flop_counts
        for event in out_profile["traceEvents"]:
            if event["name"].startswith("aten::mm"):
                self.assertEqual(
                    event["args"]["kernel_flops"],
                    flop_counts["Global"][torch.ops.aten.mm],
                )
            if event["name"].startswith("aten::addmm"):
                self.assertEqual(
                    event["args"]["kernel_flops"],
                    flop_counts["Global"][torch.ops.aten.addmm],
                )
            if event["name"].startswith("aten::baddbmm"):
                self.assertEqual(
                    event["args"]["kernel_flops"],
                    flop_counts["Global"][torch.ops.aten.baddbmm],
                )
            if event["name"].startswith("aten::bmm"):
                self.assertEqual(
                    event["args"]["kernel_flops"],
                    flop_counts["Global"][torch.ops.aten.bmm],
                )
            # TODO fix convolution
            # TODO transposed convolution
            # if event['name'].startswith('aten::convolution'):
            #     print(event['args']['kernel_flops'])
            #     self.assertEqual(event['args']['kernel_flops'], flop_counts['Global'][torch.ops.aten.convolution])
            # self.assertEqual(out_profile['traceEvents'][i]['args']['kernel_flops'], expected_flops[i])


instantiate_device_type_tests(TestAnalysis, globals())

if __name__ == "__main__":
    run_tests()
