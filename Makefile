PY ?= .venv/bin/python

.PHONY: venv test bench bench-quick e2e serve-mock all clean

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q

# One-command reproduction: synthesises audio with macOS `say`, runs the
# simulator benchmark + the real-time websocket run, writes results/ and the README table.
bench:
	$(PY) -m bench.run
	$(PY) -m bench.e2e

bench-quick:
	$(PY) -m bench.run --quick

e2e:
	$(PY) -m bench.e2e

serve-mock:
	$(PY) -m novabridge.server --backend mock --port 8080

all: test bench

clean:
	rm -rf .cache results_quick results_dev .pytest_cache
