# Project Blokkos: Receipt Detection and OCR

## Installation

---

## Using the server without docker

```bash
python3 app/app.py
```

After the command the API server will listening on *localhost:3000*

---

## Testing the project:

The `test.py` inside the *app* directory is used for testing.

```bash
python3 app/test.py <arg1> <arg2> ... <argN>
```

All arguments are **optional**, when no argument is passed, then the program will test all files from the default directory. (**TODO set default directories to a specific one**)

Available arguments:

* `[-S, --single] <image-name>`: Test a specific image by name, which is inside the testing directory.
* `--test-path <directory-path>`: Set path to the testing directory.
* `--labels-path <labels-file-path>`: Set path to the file, which contains the labels.
* `--regression-path <regression-file-path>`: Set path to the file, which contains the regression labels.
* `[-R, --test-regression]`: If specified, then regression test will also be performed on the testing directory.
* `--gpu`: To use the available GPU of the system.
* `--gen-regression`: To generate a regression test file from the results of the testing.

## Examples

Default test all with regression on gpu.
```bash
python app/test.py -R --gpu
```

Test one image on gpu
```
pytohn app/test.py --gpu -S <image-path>
```

© gaborpelesz 2020