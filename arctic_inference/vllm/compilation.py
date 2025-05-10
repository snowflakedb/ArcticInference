from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch
from vllm.compilation.backends import PiecewiseBackend, logger
from vllm.compilation.counter import compilation_counter
from vllm.utils import weak_ref_tensors

from arctic_inference.patching import ArcticPatch


class PiecewiseBackendPatch(ArcticPatch[PiecewiseBackend]):

    _orig_init = PiecewiseBackend.__init__

    def __init__(self, *args, **kwargs):
        PiecewiseBackendPatch._orig_init(self, *args, **kwargs)
        if len(self.sym_shape_indices) > 1:
            logger.warning(
                "Cudagraph is disabled for subgraph %s because it has "
                "multiple symbolic shapes", self.piecewise_compile_index)
            for entry in self.concrete_size_entries.values():
                entry.use_cudagraph = False

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_caputured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        new_input_addresses = [a for a in args if isinstance(a, torch.Tensor)]
        for a, b in zip(new_input_addresses, entry.input_addresses):
            if a.data_ptr() != b.data_ptr():
                # the input data is different, we need to copy it
                assert a.numel() <= b.numel()
                b.view(-1)[:a.numel()].copy_(a.view(-1))

        # if self.is_debugging_mode:
        #     # check if the input addresses are the same
        #     new_input_addresses = [
        #         x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        #     ]
        #     assert new_input_addresses == entry.input_addresses, (
        #         "Input addresses for cudagraphs are different during replay."
        #         f" Expected {entry.input_addresses}, got {new_input_addresses}"
        #     )

        entry.cudagraph.replay()
        return entry.output