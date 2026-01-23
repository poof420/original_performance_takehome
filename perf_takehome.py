"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict, deque
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.slot_counts = defaultdict(int)
        self.bundle_count = 0
        self.enable_slot_stats = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})
        self.slot_counts[engine] += 1
        self.bundle_count += 1

    def emit_bundle(self, **engine_slots: list[tuple]):
        instr = {
            engine: slots
            for engine, slots in engine_slots.items()
            if slots is not None and len(slots) > 0
        }
        if not instr:
            return
        for engine, slots in instr.items():
            assert len(slots) <= SLOT_LIMITS[engine]
            self.slot_counts[engine] += len(slots)
        self.bundle_count += 1
        self.instrs.append(instr)

    def slot_summary(self):
        """
        Return a minimal slot accounting summary for schedule tuning.
        """
        if not self.bundle_count:
            return {}
        summary = dict(self.slot_counts)
        summary["bundles"] = self.bundle_count
        summary["lb_load_cycles"] = (summary.get("load", 0) + SLOT_LIMITS["load"] - 1) // SLOT_LIMITS["load"]
        summary["lb_valu_cycles"] = (summary.get("valu", 0) + SLOT_LIMITS["valu"] - 1) // SLOT_LIMITS["valu"]
        summary["lb_store_cycles"] = (summary.get("store", 0) + SLOT_LIMITS["store"] - 1) // SLOT_LIMITS["store"]
        summary["lb_flow_cycles"] = (summary.get("flow", 0) + SLOT_LIMITS["flow"] - 1) // SLOT_LIMITS["flow"]
        return summary

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name=name, length=VLEN)

    def vector_const(self, val, name=None):
        scalar_addr = self.scratch_const(val, name=name)
        vec_addr = self.alloc_vec(name=None if name is None else f"{name}_vec")
        self.emit_bundle(valu=[("vbroadcast", vec_addr, scalar_addr)])
        return vec_addr

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation using VLIW scheduling.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))
        scalar_one = self.scratch_const(1, name="one_scalar")
        vec_two = self.vector_const(2, name="two")
        vec_one = self.vector_const(1, name="one")
        vec_c1 = self.vector_const(0x7ED55D16, name="hash_c1")
        vec_c2 = self.vector_const(0xC761C23C, name="hash_c2")
        vec_c3 = self.vector_const(0x165667B1, name="hash_c3")
        vec_c4 = self.vector_const(0xD3A2646C, name="hash_c4")
        vec_c5 = self.vector_const(0xFD7046C5, name="hash_c5")
        vec_c6 = self.vector_const(0xB55A4F09, name="hash_c6")
        vec_m1 = self.vector_const(4097, name="hash_m1")
        vec_m3 = self.vector_const(33, name="hash_m3")
        vec_m5 = self.vector_const(9, name="hash_m5")
        vec_s2 = self.vector_const(19, name="hash_s2")
        vec_s4 = self.vector_const(9, name="hash_s4")
        vec_s6 = self.vector_const(16, name="hash_s6")

        vec_forest_base = self.alloc_vec("forest_values_p_vec")
        self.emit_bundle(
            valu=[("vbroadcast", vec_forest_base, self.scratch["forest_values_p"])]
        )
        addr_bias = self.alloc_scratch("addr_bias")
        self.emit_bundle(
            alu=[("-", addr_bias, scalar_one, self.scratch["forest_values_p"])]
        )
        vec_addr_bias = self.alloc_vec("addr_bias_vec")
        self.emit_bundle(valu=[("vbroadcast", vec_addr_bias, addr_bias)])

        node0_scalar = self.alloc_scratch("node0_scalar")
        node0_vec = self.alloc_vec("node0_vec")
        self.emit_bundle(load=[("load", node0_scalar, self.scratch["forest_values_p"])])
        self.emit_bundle(valu=[("vbroadcast", node0_vec, node0_scalar)])

        addr_tmp = self.alloc_scratch("addr_tmp")
        node1_scalar = self.alloc_scratch("node1_scalar")
        node2_scalar = self.alloc_scratch("node2_scalar")
        self.emit_bundle(
            alu=[("+", addr_tmp, self.scratch["forest_values_p"], scalar_one)]
        )
        self.emit_bundle(load=[("load", node1_scalar, addr_tmp)])
        self.emit_bundle(alu=[("+", addr_tmp, addr_tmp, scalar_one)])
        self.emit_bundle(load=[("load", node2_scalar, addr_tmp)])
        diff12_scalar = self.alloc_scratch("diff12_scalar")
        self.emit_bundle(
            alu=[
                ("-", diff12_scalar, node2_scalar, node1_scalar),
            ]
        )
        node1_vec = self.alloc_vec("node1_vec")
        diff12_vec = self.alloc_vec("diff12_vec")
        self.emit_bundle(
            valu=[
                ("vbroadcast", node1_vec, node1_scalar),
                ("vbroadcast", diff12_vec, diff12_scalar),
            ]
        )

        idx_addrs = []
        val_addrs = []
        addr_vecs = []
        val_vecs = []
        node_val_vecs = []
        tmp_vecs = []
        for chunk in range(0, batch_size, VLEN):
            idx_addr = self.alloc_scratch(f"idx_addr_{chunk}")
            val_addr = self.alloc_scratch(f"val_addr_{chunk}")
            addr_vec = self.alloc_vec(f"addr_vec_{chunk}")
            val_vec = self.alloc_vec(f"val_vec_{chunk}")
            node_val_vec = self.alloc_vec(f"node_val_{chunk}")
            tmp_vec = self.alloc_vec(f"tmp_vec_{chunk}")
            idx_addrs.append(idx_addr)
            val_addrs.append(val_addr)
            addr_vecs.append(addr_vec)
            val_vecs.append(val_vec)
            node_val_vecs.append(node_val_vec)
            tmp_vecs.append(tmp_vec)

        for chunk_index, chunk in enumerate(range(0, batch_size, VLEN)):
            offset = self.scratch_const(chunk)
            self.emit_bundle(
                alu=[
                    ("+", idx_addrs[chunk_index], self.scratch["inp_indices_p"], offset),
                    ("+", val_addrs[chunk_index], self.scratch["inp_values_p"], offset),
                ]
            )
            self.emit_bundle(
                load=[
                    ("vload", addr_vecs[chunk_index], idx_addrs[chunk_index]),
                    ("vload", val_vecs[chunk_index], val_addrs[chunk_index]),
                ]
            )
            self.emit_bundle(
                valu=[
                    (
                        "+",
                        addr_vecs[chunk_index],
                        addr_vecs[chunk_index],
                        vec_forest_base,
                    )
                ]
            )

        n_chunks = len(addr_vecs)
        def compute_ops(val_vec, addr_vec, node_vec, tmp_vec):
            return [
                ("^", val_vec, val_vec, node_vec),
                ("multiply_add", val_vec, val_vec, vec_m1, vec_c1),
                (">>", tmp_vec, val_vec, vec_s2),
                ("^", val_vec, val_vec, vec_c2),
                ("^", val_vec, val_vec, tmp_vec),
                ("multiply_add", val_vec, val_vec, vec_m3, vec_c3),
                ("<<", tmp_vec, val_vec, vec_s4),
                ("+", val_vec, val_vec, vec_c4),
                ("^", val_vec, val_vec, tmp_vec),
                ("multiply_add", val_vec, val_vec, vec_m5, vec_c5),
                (">>", tmp_vec, val_vec, vec_s6),
                ("^", val_vec, val_vec, vec_c6),
                ("^", val_vec, val_vec, tmp_vec),
                ("&", tmp_vec, val_vec, vec_one),
                ("multiply_add", addr_vec, addr_vec, vec_two, vec_addr_bias),
                ("+", addr_vec, addr_vec, tmp_vec),
            ]

        def depth1_ops(chunk_index: int):
            ops = [
                (
                    "&",
                    tmp_vecs[chunk_index],
                    addr_vecs[chunk_index],
                    vec_one,
                ),
                (
                    "multiply_add",
                    node_val_vecs[chunk_index],
                    tmp_vecs[chunk_index],
                    diff12_vec,
                    node1_vec,
                ),
            ]
            ops.extend(
                compute_ops(
                    val_vecs[chunk_index],
                    addr_vecs[chunk_index],
                    node_val_vecs[chunk_index],
                    tmp_vecs[chunk_index],
                )
            )
            return ops

        reset_period = forest_height + 1
        round_index = [0] * n_chunks
        compute_idx = [0] * n_chunks
        load_idx = [0] * n_chunks
        per_chunk_ops = [None] * n_chunks

        load_queue = deque()
        compute_queue = deque()
        store_queue = deque()
        current_load_chunk = None
        done_count = 0
        store_count = 0

        for chunk in range(n_chunks):
            per_chunk_ops[chunk] = compute_ops(
                val_vecs[chunk],
                addr_vecs[chunk],
                node0_vec,
                tmp_vecs[chunk],
            )
            compute_queue.append(chunk)

        while True:
            valu_ops = []
            load_ops = []
            store_ops = []
            load_done = []
            compute_done = []

            if compute_queue:
                scheduled_chunks = []
                while compute_queue and len(valu_ops) < SLOT_LIMITS["valu"]:
                    chunk = compute_queue.popleft()
                    op_idx = compute_idx[chunk]
                    valu_ops.append(per_chunk_ops[chunk][op_idx])
                    compute_idx[chunk] = op_idx + 1
                    if compute_idx[chunk] < len(per_chunk_ops[chunk]):
                        scheduled_chunks.append(chunk)
                    else:
                        compute_done.append(chunk)
                compute_queue.extend(scheduled_chunks)

            if current_load_chunk is None and load_queue:
                current_load_chunk = load_queue.popleft()
            if current_load_chunk is not None:
                offset = load_idx[current_load_chunk]
                if offset < VLEN:
                    load_ops.append(
                        (
                            "load_offset",
                            node_val_vecs[current_load_chunk],
                            addr_vecs[current_load_chunk],
                            offset,
                        )
                    )
                    load_idx[current_load_chunk] += 1
                if load_idx[current_load_chunk] < VLEN and len(load_ops) < SLOT_LIMITS["load"]:
                    offset = load_idx[current_load_chunk]
                    load_ops.append(
                        (
                            "load_offset",
                            node_val_vecs[current_load_chunk],
                            addr_vecs[current_load_chunk],
                            offset,
                        )
                    )
                    load_idx[current_load_chunk] += 1
                if load_idx[current_load_chunk] >= VLEN:
                    load_done.append(current_load_chunk)
                    current_load_chunk = None

            while store_queue and len(store_ops) < SLOT_LIMITS["store"]:
                chunk = store_queue.popleft()
                store_ops.append(("vstore", val_addrs[chunk], val_vecs[chunk]))
                store_count += 1

            if valu_ops or load_ops or store_ops:
                self.emit_bundle(valu=valu_ops, load=load_ops, store=store_ops)

            pending_compute = []
            pending_load = []
            pending_store = []

            for chunk in load_done:
                round_i = round_index[chunk]
                reset_idx = (round_i + 1) % reset_period == 0
                per_chunk_ops[chunk] = compute_ops(
                    val_vecs[chunk],
                    addr_vecs[chunk],
                    node_val_vecs[chunk],
                    tmp_vecs[chunk],
                )
                if reset_idx:
                    per_chunk_ops[chunk].append(
                        ("vbroadcast", addr_vecs[chunk], self.scratch["forest_values_p"])
                    )
                compute_idx[chunk] = 0
                pending_compute.append(chunk)

            for chunk in compute_done:
                round_index[chunk] += 1
                if round_index[chunk] >= rounds:
                    done_count += 1
                    pending_store.append(chunk)
                    continue
                if round_index[chunk] % reset_period == 0:
                    reset_idx = (round_index[chunk] + 1) % reset_period == 0
                    per_chunk_ops[chunk] = compute_ops(
                        val_vecs[chunk],
                        addr_vecs[chunk],
                        node0_vec,
                        tmp_vecs[chunk],
                    )
                    if reset_idx:
                        per_chunk_ops[chunk].append(
                            ("vbroadcast", addr_vecs[chunk], self.scratch["forest_values_p"])
                        )
                    compute_idx[chunk] = 0
                    pending_compute.append(chunk)
                elif round_index[chunk] % reset_period == 1:
                    per_chunk_ops[chunk] = depth1_ops(chunk)
                    compute_idx[chunk] = 0
                    pending_compute.append(chunk)
                else:
                    pending_load.append(chunk)

            for chunk in pending_compute:
                compute_queue.append(chunk)
            for chunk in pending_load:
                load_idx[chunk] = 0
                load_queue.append(chunk)
            for chunk in pending_store:
                store_queue.append(chunk)

            if (
                done_count == n_chunks
                and store_count == n_chunks
                and not load_queue
                and current_load_chunk is None
                and not compute_queue
                and not store_queue
            ):
                break
            if not (valu_ops or load_ops or store_ops):
                raise RuntimeError("Scheduler stalled without progress")

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    slot_stats: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)
    if slot_stats:
        print("SLOT SUMMARY:", kb.slot_summary())

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    machine.run()
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
