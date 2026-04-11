# memory_lab instructions

Project style:

* Keep the project small, readable, and research-oriented.
* Prefer plain Python over abstractions.
* Heavily comment code with intent:

  * what this does
  * why it exists
  * what assumption it is making

Research rule:

* Treat memory as layered:

  * short-term raw memory
  * optional durable full memory
  * lightweight compressed stubs for low-salience items
* Before forgetting a low-salience item, prefer creating a retrieval stub instead of deleting it outright.

Scope rules:

* Do not add embeddings, learned models, APIs, vector databases, notebooks, or UI unless explicitly requested.
* Keep experiments easy to inspect from terminal output.

Current priority:

* Implement and evaluate stub memory as:
  "forget the detail, keep the address."
