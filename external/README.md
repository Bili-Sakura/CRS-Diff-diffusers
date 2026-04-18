# External dependencies

This folder is reserved for vendored external libraries.

## Diffusers

To use a local clone of Hugging Face diffusers as an external in-repo library, clone it to:

`external/diffusers`

Expected import root:

`external/diffusers/src`

If this path exists, `src/diffusers_impl/*` will prioritize it before site-packages.
