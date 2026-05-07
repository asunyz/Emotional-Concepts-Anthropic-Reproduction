# Consistency analysis report

Source: `runs/cognitive_v3_full/extractions`

Cross-method layer: 30

Cross-layer method: methodC_incontext


## B. Cross-method consistency

Higher cosine = more agreement between methods on what 'curious' (etc.) means.


Method labels: A = v2-style whole-story, B = paragraph isolation, C = paragraph in-context, D = within-stage contrast.


| concept | A vs B | A vs C | A vs D | B vs C | B vs D | C vs D |
|---|---|---|---|---|---|---|
| curious | +0.515 | +0.138 | +0.708 | +0.603 | +0.711 | +0.347 |
| uncertain | +0.311 | +0.287 | +0.393 | +0.724 | +0.814 | +0.555 |
| confident | +0.416 | +0.283 | +0.568 | +0.703 | +0.767 | +0.526 |
| surprised | +0.679 | +0.668 | +0.891 | +0.851 | +0.791 | +0.754 |
| bored | +0.517 | +0.555 | +0.883 | +0.689 | +0.572 | +0.619 |
| stubborn | +0.704 | +0.627 | +0.922 | +0.826 | +0.790 | +0.666 |
| enlightened | +0.656 | +0.650 | +0.924 | +0.806 | +0.823 | +0.767 |
| confused | +0.728 | +0.691 | +0.946 | +0.771 | +0.788 | +0.668 |
| confirmed | +0.653 | +0.656 | +0.870 | +0.755 | +0.779 | +0.764 |

## D. Cross-layer consistency (Method C)

| concept | L10-L20 | L10-L30 | L10-L36 | L20-L30 | L20-L36 | L30-L36 |
|---|---|---|---|---|---|---|
| curious | +0.799 | +0.705 | +0.553 | +0.822 | +0.660 | +0.838 |
| uncertain | +0.797 | +0.684 | +0.486 | +0.811 | +0.608 | +0.826 |
| confident | +0.815 | +0.712 | +0.543 | +0.836 | +0.647 | +0.829 |
| surprised | +0.796 | +0.698 | +0.583 | +0.809 | +0.668 | +0.806 |
| bored | +0.710 | +0.625 | +0.507 | +0.781 | +0.598 | +0.780 |
| stubborn | +0.774 | +0.706 | +0.585 | +0.824 | +0.693 | +0.860 |
| enlightened | +0.826 | +0.727 | +0.481 | +0.845 | +0.589 | +0.789 |
| confused | +0.779 | +0.649 | +0.538 | +0.787 | +0.636 | +0.834 |
| confirmed | +0.775 | +0.687 | +0.560 | +0.846 | +0.688 | +0.853 |