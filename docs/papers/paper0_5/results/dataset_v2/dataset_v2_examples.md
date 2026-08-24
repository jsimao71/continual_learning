# Dataset V2 examples

Tokens are integer symbols; `51` designates the predictive pattern and `52` is the query marker.

## balanced_ngram3

- rule: `modular_sum3`
- inputs: `[0, 0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 4, 8, 12, 52]`

## balanced_pair

- rule: `latin_square_mod4`
- inputs: `[0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 4, 8, 52]`

## functor

- rule: `functor_add_mod4`
- inputs: `[0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 40, 4, 8, 52]`

## nested_override

- rule: `nested_mod4`
- inputs: `[0, 0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 4, 8, 12, 52]`

## nested_override_control

- rule: `nested_mod4_override`
- inputs: `[1, 0, 0]` -> target `21`
- nuisance: `N6` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 5, 8, 12, 52]`

## nested_short

- rule: `nested_short_mod4`
- inputs: `[0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 8, 12, 52]`

## sparse_pair

- rule: `sparse_latin_square_mod4`
- inputs: `[0, 0]` -> target `20`
- nuisance: `N0` / `[]`
- tokens: `[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 4, 50, 50, 8, 52]`
