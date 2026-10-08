# MiMo-V2.6-Flash serving image: the pinned TensorFold 0.6.0 CUDA runtime, exllamav3 (EXL3 reference forward and
# quantiser), and TensorFold python-0.6 + this recipe's patches on the import path ahead of the base image's own.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# exllamav3 first, so a TensorFold patch change does not recompile its 175 CUDA sources. --no-deps: the base's
# PyTorch is an NVIDIA prerelease build that pip must never replace.
ARG EXL3_REPO=https://github.com/turboderp-org/exllamav3
ARG EXL3_COMMIT
ARG MAX_JOBS=12
RUN git clone -q "${EXL3_REPO}" /opt/exllamav3 && git -C /opt/exllamav3 checkout -q "${EXL3_COMMIT}" \
 && cd /opt/exllamav3 && TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=${MAX_JOBS} \
    pip install --no-deps --no-build-isolation --no-cache-dir . \
 && pip install --no-deps --no-cache-dir marisa_trie llguidance \
 && python3 -c "import exllamav3, exllamav3_ext; from importlib.metadata import version; print('exllamav3', version('exllamav3'))"

COPY src /opt/tensorfold-mimo/src
COPY tools /opt/tensorfold-mimo/tools
ENV PYTHONPATH=/opt/tensorfold-mimo/src
ARG TF_COMMIT=unknown
LABEL org.opencontainers.image.title="mimo-tf" org.opencontainers.image.revision="${TF_COMMIT}" \
      io.exllamav3.revision="${EXL3_COMMIT}"
WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "tensorfold"]
