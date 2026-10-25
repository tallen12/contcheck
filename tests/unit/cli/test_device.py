from contcheck.cli.device import CPU, GpuInfo, choose_device, describe

DISCRETE = GpuInfo(0, "AMD Radeon RX 7900 XT", "gfx1100", 20 * 2**30, integrated=False)
INTEGRATED = GpuInfo(1, "AMD Ryzen 7 9700X", "gfx1036", 30 * 2**30, integrated=True)


def test_explicit_device_wins() -> None:
    assert choose_device("cuda:1", [DISCRETE]) == "cuda:1"


def test_integrated_gpus_are_skipped_even_with_more_memory() -> None:
    assert choose_device(None, [DISCRETE, INTEGRATED]) == "cuda:0"


def test_largest_discrete_gpu_is_chosen() -> None:
    bigger = GpuInfo(2, "big", "gfx1100", 24 * 2**30, integrated=False)

    assert choose_device(None, [DISCRETE, bigger]) == "cuda:2"


def test_cpu_without_discrete_gpus() -> None:
    assert choose_device(None, [INTEGRATED]) == CPU
    assert choose_device(None, []) == CPU


def test_describe_names_the_gpu() -> None:
    assert (
        describe("cuda:0", [DISCRETE]) == "device cuda:0: AMD Radeon RX 7900 XT (gfx1100, 20 GiB)"
    )
    assert describe("cpu", [DISCRETE]) == "device cpu"
