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

from collections import defaultdict
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
        self.instrs.append(instr)

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
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
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
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
        self.add("flow", ("pause",))
        zero_const = self.scratch_const(0)
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

        addr_tmp = self.alloc_scratch("addr_tmp")
        node0_scalar = self.alloc_scratch("node0_scalar")
        node0_vec = self.alloc_vec("node0_vec")
        self.emit_bundle(
            alu=[("+", addr_tmp, self.scratch["forest_values_p"], zero_const)]
        )
        self.emit_bundle(load=[("load", node0_scalar, addr_tmp)])
        self.emit_bundle(valu=[("vbroadcast", node0_vec, node0_scalar)])
        idx_addrs = []
        val_addrs = []
        idx_vecs = []
        val_vecs = []
        for chunk in range(0, batch_size, VLEN):
            idx_addr = self.alloc_scratch(f"idx_addr_{chunk}")
            val_addr = self.alloc_scratch(f"val_addr_{chunk}")
            idx_vec = self.alloc_vec(f"idx_vec_{chunk}")
            val_vec = self.alloc_vec(f"val_vec_{chunk}")
            idx_addrs.append(idx_addr)
            val_addrs.append(val_addr)
            idx_vecs.append(idx_vec)
            val_vecs.append(val_vec)

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
                    ("vload", idx_vecs[chunk_index], idx_addrs[chunk_index]),
                    ("vload", val_vecs[chunk_index], val_addrs[chunk_index]),
                ]
            )

        n_chunks = len(idx_vecs)
        group_size = SLOT_LIMITS["valu"]
        n_buffers = 2
        group_node_addr = []
        group_node_val = []
        group_tmp1 = []
        group_tmp2 = []
        for buf_index in range(n_buffers):
            addr_buf = []
            val_buf = []
            tmp1_buf = []
            tmp2_buf = []
            for group_index in range(group_size):
                addr_buf.append(self.alloc_vec(f"node_addr_b{buf_index}_{group_index}"))
                val_buf.append(self.alloc_vec(f"node_val_b{buf_index}_{group_index}"))
                tmp1_buf.append(self.alloc_vec(f"vec_tmp1_b{buf_index}_{group_index}"))
                tmp2_buf.append(self.alloc_vec(f"vec_tmp2_b{buf_index}_{group_index}"))
            group_node_addr.append(addr_buf)
            group_node_val.append(val_buf)
            group_tmp1.append(tmp1_buf)
            group_tmp2.append(tmp2_buf)

        def emit_valu_batches(ops: list[tuple]):
            idx = 0
            while idx < len(ops):
                self.emit_bundle(valu=ops[idx : idx + SLOT_LIMITS["valu"]])
                idx += SLOT_LIMITS["valu"]

        def compute_ops(val_vec, idx_vec, node_vec, tmp1_vec, tmp2_vec):
            return [
                ("^", val_vec, val_vec, node_vec),
                ("multiply_add", val_vec, val_vec, vec_m1, vec_c1),
                (">>", tmp1_vec, val_vec, vec_s2),
                ("^", tmp2_vec, val_vec, vec_c2),
                ("^", val_vec, tmp1_vec, tmp2_vec),
                ("multiply_add", val_vec, val_vec, vec_m3, vec_c3),
                ("+", tmp1_vec, val_vec, vec_c4),
                ("<<", tmp2_vec, val_vec, vec_s4),
                ("^", val_vec, tmp1_vec, tmp2_vec),
                ("multiply_add", val_vec, val_vec, vec_m5, vec_c5),
                (">>", tmp1_vec, val_vec, vec_s6),
                ("^", tmp2_vec, val_vec, vec_c6),
                ("^", val_vec, tmp1_vec, tmp2_vec),
                ("*", idx_vec, idx_vec, vec_two),
                ("&", tmp1_vec, val_vec, vec_one),
                ("+", idx_vec, idx_vec, vec_one),
                ("+", idx_vec, idx_vec, tmp1_vec),
            ]

        reset_period = forest_height + 1
        def take_block_ops(block_ops, block_positions, max_ops):
            valu_ops = []
            for block_index in range(len(block_ops)):
                if len(valu_ops) >= max_ops:
                    break
                if block_positions[block_index] >= len(block_ops[block_index]):
                    continue
                valu_ops.append(block_ops[block_index][block_positions[block_index]])
                block_positions[block_index] += 1
            return valu_ops

        def emit_valu_with_prev(current_ops, prev_ops, prev_positions):
            idx = 0
            while idx < len(current_ops):
                valu_ops = []
                while idx < len(current_ops) and len(valu_ops) < SLOT_LIMITS["valu"]:
                    valu_ops.append(current_ops[idx])
                    idx += 1
                if prev_ops is not None and len(valu_ops) < SLOT_LIMITS["valu"]:
                    valu_ops.extend(
                        take_block_ops(
                            prev_ops,
                            prev_positions,
                            SLOT_LIMITS["valu"] - len(valu_ops),
                        )
                    )
                self.emit_bundle(valu=valu_ops)

        for round_index in range(rounds):
            if round_index % reset_period == 0:
                for block_start in range(0, n_chunks, group_size):
                    block_end = min(block_start + group_size, n_chunks)
                    block_chunks = list(range(block_start, block_end))
                    block_ops = []
                    for block_slot, chunk_index in enumerate(block_chunks):
                        block_ops.append(
                            compute_ops(
                                val_vecs[chunk_index],
                                idx_vecs[chunk_index],
                                node0_vec,
                                group_tmp1[0][block_slot],
                                group_tmp2[0][block_slot],
                            )
                        )
                    block_positions = [0] * len(block_chunks)
                    while True:
                        valu_ops = take_block_ops(
                            block_ops, block_positions, SLOT_LIMITS["valu"]
                        )
                        if not valu_ops:
                            break
                        self.emit_bundle(valu=valu_ops)
                if (round_index + 1) % reset_period == 0:
                    reset_ops = [
                        (
                            "^",
                            idx_vecs[chunk_index],
                            idx_vecs[chunk_index],
                            idx_vecs[chunk_index],
                        )
                        for chunk_index in range(n_chunks)
                    ]
                    emit_valu_batches(reset_ops)
                continue
            prev_ops = None
            prev_positions = None
            for block_index, block_start in enumerate(range(0, n_chunks, group_size)):
                block_end = min(block_start + group_size, n_chunks)
                block_chunks = list(range(block_start, block_end))
                buf = block_index % n_buffers
                node_addr_ops = []
                for block_slot, chunk_index in enumerate(block_chunks):
                    node_addr_ops.append(
                        (
                            "+",
                            group_node_addr[buf][block_slot],
                            idx_vecs[chunk_index],
                            vec_forest_base,
                        )
                    )
                emit_valu_with_prev(node_addr_ops, prev_ops, prev_positions)

                for block_slot in range(len(block_chunks)):
                    for stage in range(4):
                        valu_ops = []
                        if prev_ops is not None:
                            valu_ops = take_block_ops(
                                prev_ops, prev_positions, SLOT_LIMITS["valu"]
                            )
                        self.emit_bundle(
                            load=[
                                (
                                    "load_offset",
                                    group_node_val[buf][block_slot],
                                    group_node_addr[buf][block_slot],
                                    stage * 2,
                                ),
                                (
                                    "load_offset",
                                    group_node_val[buf][block_slot],
                                    group_node_addr[buf][block_slot],
                                    stage * 2 + 1,
                                ),
                            ],
                            valu=valu_ops,
                        )

                block_ops = []
                for block_slot, chunk_index in enumerate(block_chunks):
                    block_ops.append(
                        compute_ops(
                            val_vecs[chunk_index],
                            idx_vecs[chunk_index],
                            group_node_val[buf][block_slot],
                            group_tmp1[buf][block_slot],
                            group_tmp2[buf][block_slot],
                        )
                    )
                if prev_ops is not None:
                    while True:
                        valu_ops = take_block_ops(
                            prev_ops, prev_positions, SLOT_LIMITS["valu"]
                        )
                        if not valu_ops:
                            break
                        self.emit_bundle(valu=valu_ops)

                prev_ops = block_ops
                prev_positions = [0] * len(block_chunks)

            if prev_ops is not None:
                while True:
                    valu_ops = take_block_ops(
                        prev_ops, prev_positions, SLOT_LIMITS["valu"]
                    )
                    if not valu_ops:
                        break
                    self.emit_bundle(valu=valu_ops)

            if (round_index + 1) % reset_period == 0:
                reset_ops = [
                    ("^", idx_vecs[chunk_index], idx_vecs[chunk_index], idx_vecs[chunk_index])
                    for chunk_index in range(n_chunks)
                ]
                emit_valu_batches(reset_ops)

        for chunk_index in range(n_chunks):
            self.emit_bundle(load=[("const", addr_tmp, chunk_index * VLEN)])
            self.emit_bundle(
                alu=[
                    (
                        "+",
                        addr_tmp,
                        self.scratch["inp_values_p"],
                        addr_tmp,
                    )
                ]
            )
            self.emit_bundle(store=[("vstore", addr_tmp, val_vecs[chunk_index])])
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
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
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

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
