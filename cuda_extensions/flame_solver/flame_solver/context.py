from dataclasses import dataclass


@dataclass(slots=True)
class SolverContext:
    cublas_handle: object | None = None
    cusolver_handle: object | None = None

    @classmethod
    def create(cls, cuda_ext):
        cublas_handle = cuda_ext.create_cublas_handle()
        cusolver_handle = cuda_ext.create_cusolver_handle()
        return cls(cublas_handle=cublas_handle, cusolver_handle=cusolver_handle)

    @property
    def available(self) -> bool:
        return self.cublas_handle is not None and self.cusolver_handle is not None
