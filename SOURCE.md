# Source and attribution

ReLoCache separates PIC serving components from an earlier SemPIC research
implementation. The training algorithm and research evaluation code are maintained
in [SemPIC](https://github.com/jn12-29/SemPIC).
Training, datasets, and unpublished research artifacts are not
required to install or run the service.

## vLLM

- Upstream: https://github.com/vllm-project/vllm
- Base commit: `08b3e67b669d18d5bbd14e456b8bb4794916e402`
- Fork: https://github.com/jn12-29/ReLoCache-vLLM
- License: Apache-2.0; upstream notices and file attributions are retained.

The fork starts at the real upstream commit, tagged `upstream-base`, with a
shallow local history boundary. Later commits contain the execution extensions
used by ReLoCache. The parent repository pins the fork as `third_party/vllm`.
Review the cumulative changes with:

```bash
git -C third_party/vllm diff upstream-base..HEAD
```

PIC semantics, kernels, adapters, and integration tests live in ReLoCache. The
fork retains native execution and narrow extension points; see
[the patch boundary](docs/vllm-patch.md).

## Browser dependencies

The package bundles Marked 18.1.1 (MIT) and DOMPurify 3.4.17
(Apache-2.0 or MPL-2.0). Their source headers and license texts are retained in
`src/relocache/webchat_assets/THIRD_PARTY_LICENSES.txt` and the bundled modules.
The demo handbook is fictional example content.

Model weights, trained adapters, and datasets are not distributed with this
repository. Obtain them separately under their respective licenses.
