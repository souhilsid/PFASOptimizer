# openLCA data mount

This folder is intentionally empty in Git.

For the real OpenLCA engine, the deployed container needs a full openLCA workspace:

```text
/app/data/
  databases/
    Biochar/
  libraries/
    ...
```

Do not commit licensed or large openLCA databases unless you have confirmed that
the data license allows cloud distribution.
