# Initialization Source

- Source repository: `/mnt/sata2/xh/EfficientAgent/sempic_iclr`.
- Source commit: `642fb31bf12c54c74c7b8970067bd4f0da6e3ac1`.
- Extracted subtree: `third_party/vllm`.
- Last source commit changing that subtree: `26799674f52032d36df61418c28ea1df5d0695df`.
- Upstream vLLM base: `08b3e67b669d18d5bbd14e456b8bb4794916e402`.

The fork starts at the real upstream commit, imported from the local upstream reference repository with a shallow history boundary and tagged `upstream-base`. The next commit consolidates the current ReLoCache execution changes, including the Native PIC extraction and subsequent service separation. Use `git -C third_party/vllm diff upstream-base..main` to review the fork against upstream. Existing upstream license and attribution files are retained.

The original repositories and environments are read-only. No environment, compiled extension, runtime cache, model weights, or experiment output was copied. No runtime validation was performed during initialization.

Future installation must create a new project environment and write build caches, service caches, logs, and exported deployment artifacts under new paths. Existing models and completed research artifacts may be consumed read-only. Keep the current upstream version while extracting the service.
