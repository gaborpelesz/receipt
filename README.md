# Project Blokkos: Receipt Detection and OCR

## Installation

Put the entire `models` folder under the app folder (app/models).  
After that run `make build`, then `make run`. Building the project can take more than 5-10 minutes.

---

## Using the server without docker (not recommended)

```bash
python3 app/app.py
```

After the command the API server will listening on *localhost:3000*

---

## Testing the project (DEBUG tool):

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
python app/test.py --gpu -S <image-path>
```

## Using the docker app server

### POST /process_receipt

The request type must be *multipart/form-data* and it must have an `image` field containing an image file. See the *postman_example* for more info.

The `image` is a full sized image, with no additional pre-processing. The vision backend will handle all the rotation, resizing and any other operation if necessary.

### Responses

All responses will be in JSON format.

Response format:

```json
{
    "status": "",
    "status_message": "",
    "receipt": {
        "AP": "",
        "date": "",
        "time": ""
    },
    "runtime": ""
}
```

or, when BAD REQUEST or catched INTERNAL SERVER ERROR

```json
{
    "status": "",
    "status_message": ""
}
```

Successful response example -> Receipt not found on the image  
status code: 404 NOT FOUND

```json
{
    "status": "Success",
    "status_message": "No receipt found on the image.",
    "receipt": {
        "AP": "?",
        "date": "?",
        "time": "?"
    },
    "runtime": "100ms"
}
```

Successful response example -> receipt found and processed
status code: 200 OK

```json
{
    "status": "Success",
    "status_message": "Successful processing of the receipt on the image.",
    "receipt": {
        "AP": "A06401234",
        "date": "2020.07.09",
        "time": "12:13"
    },
    "runtime": "1234ms"
}
```

## POSTMAN

Postman examples are under the *postman_examples* folder.  


© gaborpelesz 2020