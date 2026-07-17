# Official split cleaning: before and after

The German and English files use the same pair structure and were filtered with
identical masks. All counts below are therefore per language unless stated
otherwise.

Cleaning priority: **Test > Validation > Train**. Offer pairs are compared
without regard to left/right order. Pairs with conflicting labels were deleted
from every occurrence. Cleaning was performed independently within each
corner-case family (`20cc80`, `50cc50`, and `80cc20`).

## Overall

| Split | Rows before | Deleted | Rows after |
| --- | ---: | ---: | ---: |
| Train | 107,105 | 1,283 | 105,822 |
| Validation | 94,500 | 1,067 | 93,433 |
| Test | 40,214 | 255 | 39,959 |
| **Total per language** | **241,819** | **2,605** | **239,214** |
| **German and English combined** | **483,638** | **5,210** | **478,428** |

## Training sets

| Configuration | Size | Rows before | Deleted | Rows after | Matches before | Matches after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20cc80rnd` | small | 2,500 | 36 | 2,464 | 500 | 485 |
| `20cc80rnd` | medium | 6,000 | 87 | 5,913 | 1,500 | 1,457 |
| `20cc80rnd` | large | 27,393 | 281 | 27,112 | 13,273 | 13,100 |
| `50cc50rnd` | small | 2,500 | 39 | 2,461 | 500 | 482 |
| `50cc50rnd` | medium | 6,000 | 95 | 5,905 | 1,500 | 1,464 |
| `50cc50rnd` | large | 27,324 | 300 | 27,024 | 13,196 | 13,024 |
| `80cc20rnd` | small | 2,500 | 25 | 2,475 | 500 | 492 |
| `80cc20rnd` | medium | 6,000 | 103 | 5,897 | 1,500 | 1,462 |
| `80cc20rnd` | large | 26,888 | 317 | 26,571 | 12,896 | 12,747 |

## Test sets

Only global label conflicts were removed from test sets.

| Configuration | Rows before | Deleted | Rows after | Matches before | Matches after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20cc80rnd000un` | 4,478 | 24 | 4,454 | 408 | 407 |
| `20cc80rnd050un` | 4,455 | 26 | 4,429 | 355 | 352 |
| `20cc80rnd100un` | 4,459 | 10 | 4,449 | 366 | 360 |
| `50cc50rnd000un` | 4,470 | 43 | 4,427 | 366 | 364 |
| `50cc50rnd050un` | 4,466 | 48 | 4,418 | 355 | 346 |
| `50cc50rnd100un` | 4,479 | 21 | 4,458 | 382 | 369 |
| `80cc20rnd000un` | 4,474 | 34 | 4,440 | 346 | 342 |
| `80cc20rnd050un` | 4,470 | 33 | 4,437 | 371 | 361 |
| `80cc20rnd100un` | 4,463 | 16 | 4,447 | 356 | 348 |

## Validation sets

| Configuration | Size | Rows before | Deleted | Rows after | Matches after | Seen products |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20cc80rnd000un` | small | 2,500 | 16 | 2,484 | 491 | 100% |
| `20cc80rnd000un` | medium | 3,500 | 20 | 3,480 | 491 | 100% |
| `20cc80rnd000un` | large | 4,500 | 25 | 4,475 | 491 | 100% |
| `20cc80rnd050un` | small | 2,500 | 26 | 2,474 | 490 | 50% |
| `20cc80rnd050un` | medium | 3,500 | 31 | 3,469 | 490 | 50% |
| `20cc80rnd050un` | large | 4,500 | 38 | 4,462 | 490 | 50% |
| `20cc80rnd100un` | small | 2,500 | 26 | 2,474 | 490 | 50% |
| `20cc80rnd100un` | medium | 3,500 | 31 | 3,469 | 490 | 50% |
| `20cc80rnd100un` | large | 4,500 | 38 | 4,462 | 490 | 50% |
| `50cc50rnd000un` | small | 2,500 | 35 | 2,465 | 479 | 100% |
| `50cc50rnd000un` | medium | 3,500 | 44 | 3,456 | 479 | 100% |
| `50cc50rnd000un` | large | 4,500 | 50 | 4,450 | 479 | 100% |
| `50cc50rnd050un` | small | 2,500 | 35 | 2,465 | 487 | 50% |
| `50cc50rnd050un` | medium | 3,500 | 47 | 3,453 | 487 | 50% |
| `50cc50rnd050un` | large | 4,500 | 50 | 4,450 | 487 | 50% |
| `50cc50rnd100un` | small | 2,500 | 35 | 2,465 | 487 | 50% |
| `50cc50rnd100un` | medium | 3,500 | 47 | 3,453 | 487 | 50% |
| `50cc50rnd100un` | large | 4,500 | 50 | 4,450 | 487 | 50% |
| `80cc20rnd000un` | small | 2,500 | 30 | 2,470 | 485 | 100% |
| `80cc20rnd000un` | medium | 3,500 | 45 | 3,455 | 485 | 100% |
| `80cc20rnd000un` | large | 4,500 | 48 | 4,452 | 485 | 100% |
| `80cc20rnd050un` | small | 2,500 | 34 | 2,466 | 490 | 50% |
| `80cc20rnd050un` | medium | 3,500 | 52 | 3,448 | 490 | 50% |
| `80cc20rnd050un` | large | 4,500 | 64 | 4,436 | 490 | 50% |
| `80cc20rnd100un` | small | 2,500 | 34 | 2,466 | 490 | 50% |
| `80cc20rnd100un` | medium | 3,500 | 52 | 3,448 | 490 | 50% |
| `80cc20rnd100un` | large | 4,500 | 64 | 4,436 | 490 | 50% |

## Removal reasons per language

| Reason | Removed rows |
| --- | ---: |
| Label conflict | 632 |
| Validation overlaps a protected test pair | 898 |
| Training overlaps a protected test pair | 519 |
| Training overlaps retained validation | 556 |

## Verification

- All 90 gzip files are readable.
- German and English remain pairwise aligned.
- No internal offer-pair or `pair_id` duplicates remain.
- No label conflicts remain across the official files.
- Train, validation, and test are pair-disjoint within every corner-case family.
- Validation seen-product shares are exactly 100% for `000un` and 50% for both
  `050un` and `100un`. The `100un` designation applies to the test condition.
