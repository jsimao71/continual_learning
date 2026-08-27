# FF-only successor construction

This is a fixed-weight existence witness, not a minimality or acquisition claim.  The legal domain is A->B, B->C, C->D.  `d_model=4`, `L=1`, `H=0` effectively for the primary FF-only topology; the recorded attention matrices are zero.  Embeddings and unembedding are the four-dimensional identity.  With row-vector convention, `W1=I`, ReLU is identity on the one-hot legal inputs, and `W2=M-I`, where row `i` of `M` is the successor one-hot.  The identity residual therefore cancels and leaves exactly `M e_i`.  LayerNorm and dropout are omitted; the positional matrix is zero.

Canonical input B has embedding `[0,1,0,0]`, zero Q/K/scores/V/head update, FF preactivation `[0,1,0,0]`, and post-FF/logits `[0,0,1,0]`.  Its exact output margin is 1.  SA-only is the identity and fails all three successor cases; SA+FF passes because its SA branch is explicitly zero.
