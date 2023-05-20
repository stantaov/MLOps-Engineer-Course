## Logging

Logging is valuable for understanding the events that occur while running your program. For example, if you run your model overnight and the results the following morning are not what you expect, log messages can help you understand more about the context in those results occurred. Let's learn about the qualities that make a log message effective.

### Log Messages

Logging is the process of recording messages to describe events that have occurred while running your software. Let's take a look at a few examples, and learn tips for writing good log messages.

#### Tip: Be professional and clear

#### Tip: Be concise and use normal capitalization


#### Tip: Choose the appropriate level for logging

_Debug_: Use this level for anything that happens in the program.  _Error_: Use this level to record any error that occurs.  _Info_: Use this level to record all actions that are user driven or system specific, such as regularly scheduled operations.

#### Tip: Provide any useful information

Below is the sample logging code

```python
import logging 
logging.basicConfig(  
	filename='./results.log',  
	level=logging.INFO,  
	filemode='w',  
	format='%(name)s - %(levelname)s - %(message)s')
```
