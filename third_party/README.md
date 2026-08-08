# Third-party checkouts

Do not vendor upstream repositories, model weights or datasets into this Git repository.

The server setup script places optional checkouts here:

- `MOSS-Audio/`: official `OpenMOSS/MOSS-Audio` checkout;
- `sam-audio/`: optional official `facebookresearch/sam-audio` checkout after checkpoint access;
- model weights belong under the ignored top-level `weights/` directory.

Record upstream commit SHAs in every experiment card. The setup script does not update an existing checkout automatically, so a run never silently changes its model implementation.
