# Dataset layout

Obtain the cropped Extended Yale B images independently from the [dataset page](https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full). The original [UCSD description](https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b) provides background on the underlying collection. Check the data provider's terms before using or redistributing images.

Place PGM files in subject folders:

```text
data/YaleFace/cropped/
  yaleB11/
    yaleB11_P00A+000E+00.pgm
    ...
  yaleB12/
    ...
```

Filenames must encode subject, pose, azimuth (A), and elevation (E). The split generator skips ambient frames and filenames without angle metadata. It prints the usable image and subject counts from the copy you supply; counts can differ between cropped derivatives.

Images and generated split CSVs are intentionally excluded from version control. The checked-in fold summary is an example for one local image set, not a substitute for generating splits from your own copy.
