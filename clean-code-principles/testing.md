## Testing

Testing your code is essential before deployment. It helps you catch errors and faulty conclusions before they make any major impact. 

### Testing and Data Science

-   Problems that could occur in data science aren’t always easily detectable; you might have values being encoded incorrectly, features being used inappropriately, or unexpected data breaking assumptions.
-   To catch these errors, you have to check for the quality and accuracy of your  _analysis_  in addition to the quality of your  _code_. Proper testing is necessary to avoid unexpected surprises and have confidence in your results.
-   Test-driven development (TDD): A development process in which you write tests for tasks before you even write the code to implement those tasks.
-   Unit test: A type of test that covers a “unit” of code—usually a single function—independently from the rest of the program.

#### Resources

-   Four Ways Data Science Goes Wrong and How Test-Driven Data Analysis Can Help:  [Blog Post](https://www.predictiveanalyticsworld.com/patimes/four-ways-data-science-goes-wrong-and-how-test-driven-data-analysis-can-help/6947/)
-   Ned Batchelder: Getting Started Testing:  [Slide Deck](https://speakerdeck.com/pycon2014/getting-started-testing-by-ned-batchelder)  and  [Presentation Video](https://www.youtube.com/watch?v=FxSsnHeWQBY)

### Unit Tests

We want to test our functions in a way that is repeatable and automated. Ideally, we'd run a test program that runs all our unit tests and cleanly lets us know which ones failed and which ones succeeded. 

#### Unit Testing: Advantages and Disadvantages

The advantage of unit tests is that they are isolated from the rest of your program, and thus, no dependencies are involved. They don't require access to databases, APIs, or other external sources of information. However, passing unit tests isn’t always enough to prove that our program is working successfully. To show that all the parts of our program work with each other properly, communicating and transferring data between them correctly, we use integration tests. 

To learn more about integration testing and how integration tests relate to unit tests, see  [Integration Testing](https://www.fullstackpython.com/integration-testing.html). That article contains other very useful links as well.

### Unit Testing Tools

To install  `pytest`, run  `pip install -U pytest`  in your terminal. You can see more information on getting started  [here](https://docs.pytest.org/en/latest/getting-started.html).

-   Create a test file starting with  `test_`.
-   Define unit test functions that start with  `test_`  inside the test file.
-   Enter  `pytest`  into your terminal in the directory of your test file and it detects these tests for you.

`test_`  is the default; if you wish to change this, you can learn how in this  [`pytest`configuration](https://docs.pytest.org/en/latest/example/pythoncollection.html).

In the test output, periods represent successful unit tests and Fs represent failed unit tests. Since all you see is which test functions failed, it's wise to have only one  `assert` statement per test. Otherwise, you won't know exactly how many tests failed or which tests failed.

Your test won't be stopped by failed  `assert`  statements, but it will stop if you have syntax errors.

[Some additional resources can be found here on testing using Google's cloud-based systems](https://developers.google.com/machine-learning/testing-debugging/pipeline/deploying).

### Using Pytest Fixtures for Parameterized Tests

Pytest Fixtures are special functions that come into the picture when you need to:

-   Either write a  _parameterized_  test method, meaning you want to pass some arguments to the test method.
-   Or reuse a block of code in multiple test methods.

We define a Fixture using  `@pytest.fixture`  decorator ahead of the (non-test) function we want to repurpose in the test methods, as shown in the example below:

```python
# File: test_mylibrary.py  
# Pytest filename starts with "test_...."  
import pytest 
##################################  
""" Function to test """  
def  import_data(pth):  
	df = pd.read_csv(pth)  
	return df 
##################################  
""" Fixture - The test function test_import_data() will use the return of path() as an argument """  
@pytest.fixture(scope="module")  
def  path():  
	return  "./data/bank_data.csv"  
##################################  
""" Test method """  
def  test_import_data(path): 
	try:  df = import_data(path)  
	except FileNotFoundError as err:  
		logging.error("File not found")  
		raise err 
	# Check the df shape  
	try:  
		assert df.shape[0]  >  0  
		assert df.shape[1]  >  0 
	except AssertionError as err:  
		logging.error(  "Testing import_data: The file doesn't appear to have rows and columns")  
		raise err 
	return df 
##################################
```
Run the  _test_mylibrary.py_  file above using the  `pytest`  command in your terminal.

In the example above,

-   The  _path()_  fixture function is being used as an argument to the  _test_import_data()_method.
    
-   The  _path()_  fixture function has  `scope="module"`. You can share fixtures across classes, modules, packages or session.
    
The fixture function is reusable, meaning you can use it in multiple test functions. Note that fixture functions are not meant to be called  _directly_, but are used automatically when test methods request them as parameters.

Moreover, you can define setup/teardown statements in the fixture function that you may want to use in multiple test functions.

Note that Fixtures can request other fixtures, and there are plenty of built-in Pytest fixtures available.

### Additional Resource

-   [How to use fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html)
    
-   [Parametrizing fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixture-parametrize)  - Read this section to understand how you can pass parameters to the fixture functions using the built-in request object. In such cases, we use the fixture decorator as:
    
`@pytest.fixture(scope="module", params=["argument1",  "argument2"])`

The  `params`  list contains a list of values for each of which the fixture function will execute. You can access the  _parameters list_  later inside the fixture function as:

`value = request.param`

For example, see a fixture below that will execute for each parameter:

`@pytest.fixture(scope="module", params=["./data/bank_data.csv",  "./data/hospital_data.csv"])  def  path():   value = request.param
  return value`

### Pytest Namespace

#### Passing Values Across Test Functions

In some cases, you may want to test a series of functions dependent on each other, meaning the return of one function is used as an argument to the other function.

In such cases, you can store your test cases' results either in the Pytest Namespace or Cache.

#### Method 1: Using Namespace

conftest.py file..

> The  [conftest.py](https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files)  file serves as a means of providing fixtures for an entire directory. Any test can use fixtures defined in a conftest.py in that package

In summary, the Pytest looks for a  _conftest.py_  file when:

-   Tests from multiple test modules want to access the fixture functions
-   You want to define Pytest configurations, such as storing variables in the Namespace.

Let's see an example where you'd want to store a Dataframe from one test function and access it later in another test function.

-   The example below uses the Namespace to store and access the Dataframe object.

```python
# conftest.py  
import pytest 
def df_plugin():  
	return  None  
# Creating a Dataframe object 'pytest.df' in Namespace  
def  pytest_configure():  
	pytest.df = df_plugin()
```

-   Once you have the  _conftest.py_  above ready, you can access and redefine  `pytest.df`  in test functions as:

```python
# Test function  
# See the `pytest.df = df` statement to store the variable in Namespace  
def  test_import_data():  
	try:  df = import_data("./data/bank_data.csv")  
	except FileNotFoundError as err:  
		logging.error("File not found")  
		raise err 
		''' Some assertion statements per your requirement. '''  
	pytest.df = df 
	return df
```

-   Next, you can access the Dataframe object in the  _test_function_two()_  as:

```python
# Test function  
# See the `df = pytest.df` statement accessing the Dataframe object from Namespace  
def  test_function_two():  
	df = pytest.df 
	''' Some assertion statements per your requirement. '''
```

Useful link:  [In pytest, what is the use of conftest.py files?](https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files)

#### Method 2: Using Cache

Cache helps in saving time on repeated test runs, as well as sharing data between tests.

-   The cache object is available via  `request.config.cache.set()`  or  `request.config.cache.get()`  fixture as:

```python
# Test function 
# It uses the built-in request fixture  
def  test_import_data(request):  
	try:  df = import_data("./data/bank_data.csv")  
	except FileNotFoundError as err:  
		logging.error("File not found")  
		raise err 
		''' Some assertion statements per your requirement. '''  
	request.config.cache.set('cache_df', df)  
	return df
```

-   You can access the Dataframe object in the  _test_function_two()_  as:

```python
# Test function  
def  test_function_two(request):  
	df = request.config.cache.get('cache_df',  None)  
	''' Some assertion statements per your requirement. '''
```

Reference:  [config.cache object](https://docs.pytest.org/en/6.2.x/cache.html#the-new-config-cache-object)

### Test-Driven Development and Data Science

-   _Test-driven development_: Writing tests before you write the code that’s being tested. Your test fails at first, and you know you’ve finished implementing a task when the test passes.
-   Tests can check for different scenarios and edge cases before you even start to write your function. When start implementing your function, you can run the test to get immediate feedback on whether it works or not as you tweak your function.
-   When refactoring or adding to your code, tests help you rest assured that the rest of your code didn't break while you were making those changes. Tests also helps ensure that your function behavior is repeatable, regardless of external parameters such as hardware and time.

Test-driven development for data science is relatively new and is experiencing a lot of experimentation and breakthroughs. You can learn more about it by exploring the following resources:

-   [Data Science TDD](https://www.linkedin.com/pulse/data-science-test-driven-development-sam-savage/)
-   [TDD is Essential for Good Data Science: Here's Why](https://medium.com/@karijdempsey/test-driven-development-is-essential-for-good-data-science-heres-why-db7975a03a44)
-   [Testing Your Code](http://docs.python-guide.org/en/latest/writing/tests/)  (general python TDD)

