# Summary Results

| Model                                                 | Mean Dice |
|-------------------------------------------------------|-----------|
| __nnU-net__                                           | __0.894__ |
| U-Net 3+, basic                                       | 0.874     |
| U-Net 3+, with prioritised sampling                   | 0.873     |
| U-Net 3+, conditioned (position)                      | 0.876     |
| U-Net 3+, with depth maps                             | 0.879     |
| U-Net 3+, conditioned (probability map, 37 areas)     | 0.880     |
| U-Net 3+, conditioned (probability map, 120 areas)    | 0.881     |
| U-Net 3+, conditioned (prob map and pos)              | 0.883     |
| U-Net 3+, conditioned (prob map and pos) + depth maps | 0.883     |
| U-Net, longer training, condition + depth maps        | 0.886     |
