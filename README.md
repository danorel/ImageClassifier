# RailsReactML
Implementing the **classification** problem: **dublicate**, **modification**, **similar classification**.

**Console application**. Importing the *argparse* library to work with console parameters.
        
 ```
 python3 solution.py -h
 ```

```
usage: solution.py [-h] [-f FUNC] [-d DUB] [-m MOD] [-s SIM] src

positional arguments:
  src                   The working directory, where we store the images

optional arguments:
  -h, --help            show this help message and exit
  -f FUNC, --func FUNC  The functions we use to solve the problem: ahash,
                        dhash, phash. The default algorithm is dhash
  -d DUB, --dub DUB     Dublicate parameter. The optimal value is 0 for all
                        algorithms
  -m MOD, --mod MOD     Modification parameter. The optimal values are: 6 -
                        for ahash/phash and 14 - for dhash
  -s SIM, --sim SIM     Similar parameter. The optimal values are: 13 - for
                        ahash/phash and 32 - for dhash
```                        
