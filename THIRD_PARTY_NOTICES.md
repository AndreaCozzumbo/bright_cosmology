# Third-party code

The MIT licence of this repository (`LICENSE`) covers our own code. The following files come from
other projects.

## GWFish

`src/bright_cosmology/priors.py` and `src/bright_cosmology/_gwfish_patches.py` are modified versions
of code of [GWFish](https://github.com/janosch314/GWFish), and remain under the terms of its authors. The ET and
CE40 noise curves in `detectors/psd` are those distributed with GWFish.

- U. Dupletsa et al., *GWFish: A simulation software to evaluate parameter-estimation capabilities
  of gravitational-wave detector networks*, Astronomy and Computing 42, 100671 (2022),
  [arXiv:2205.02499](https://arxiv.org/abs/2205.02499)
- U. Dupletsa et al., *Validating Prior-informed Fisher-matrix Analyses against GWTC Data*,
  Phys. Rev. D 111, 024036 (2025), [arXiv:2404.16103](https://arxiv.org/abs/2404.16103)

## Truncated multivariate normal sampler

`src/bright_cosmology/minimax_tilting_sampler.py`, as distributed with GWFish, is taken with small
modifications from [truncated-mvn-sampler](https://github.com/brunzema/truncated-mvn-sampler), which implements

- Z. I. Botev, *The normal law under linear restrictions: simulation and estimation via minimax
  tilting*, Journal of the Royal Statistical Society Series B 79, 125-148 (2016)

Its licence is:

```
MIT License

Copyright (c) 2021 Paul Brunzema

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
