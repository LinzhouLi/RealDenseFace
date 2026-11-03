"""Numerical assertions used to validate actual ZeroGPU requests."""
import gc
import torch


def native_probe():
    import nvdiffrast.torch as dr
    from flame_solver import FlameSolverOps, SolverContext
    ops = FlameSolverOps()
    report = []
    for i in range(3):
        ctx = SolverContext.create(ops.extension)
        raster = None
        try:
            j_cpu = torch.tensor([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.], [1., -1., 1.]], dtype=torch.float64)
            r_cpu = torch.tensor([1., -2., 3., .5], dtype=torch.float64)
            h = j_cpu.T @ j_cpu
            expected = torch.linalg.solve(h + torch.diag(.01 * h.diag() + 1e-6), -j_cpu.T @ r_cpu)
            j, r = j_cpu.float().cuda(), r_cpu.float().cuda()
            actual = ops.solve_normal_equation(ctx, j, r, .01)
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-4, atol=1e-5)
            batched = ops.solve_normal_equation(ctx, j[None].repeat(2, 1, 1), r[None].repeat(2, 1), .01)
            torch.testing.assert_close(batched.cpu().double(), expected[None].repeat(2, 1), rtol=1e-4, atol=1e-5)
            raster = dr.RasterizeCudaContext()
            v = torch.tensor([[[-.7, -.7, 0., 1.], [.7, -.7, 0., 1.], [0., .7, 0., 1.]]], device="cuda")
            f = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
            rast, _ = dr.rasterize(raster, v, f, resolution=[64, 64])
            count = int((rast[..., 3] > 0).sum().item())
            assert 500 < count < 1500, count
            torch.cuda.synchronize()
            report.append({"iteration": i + 1, "solver_max_error": float((actual.cpu().double() - expected).abs().max()), "raster_pixels": count})
        finally:
            torch.cuda.synchronize()
            ops.extension.destroy_cublas_handle(ctx.cublas_handle)
            ops.extension.destroy_cusolver_handle(ctx.cusolver_handle)
            del raster
            gc.collect()
    return {"passed": True, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "tests": report}
