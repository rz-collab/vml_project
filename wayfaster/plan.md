### Shadow attacks plan

1. See performance of shadow attack. Show examples with highest change in trav map

Note that they already have this data augmentation:
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

2. Adv Training. (need to test model's change from original model, then also under adv attack)

<!-- 3. Maybe try data augmentation random shadows later, and see how it performs. -->

These attacks aren't that realistic, but physically realizable. If it works, it can show importance on shadow removal. This would test diff data augmentation vs adv training.

Notes:

- This attack isn't parallelizable over batches unfortunately (unlike PSD L_inf attacks), so it needs to generate attack for each batch separately.

### TODO

Still need to understand truly how wayfaster work by evaluating on some data.
