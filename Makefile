.PHONY: test run demo clean

test:
	python3 -m unittest discover -s tests -v

run:
	python3 -m ward.cli scenarios/example.json

demo:
	python3 -m ward.cli scenarios/alarm-storm.json --seed 42

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
