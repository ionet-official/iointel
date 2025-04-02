prepare_for_tests:
	export PORT=8080
	docker pull searxng/searxng
	docker run --rm -d \
		-p ${PORT}:8080 \
		-v "${PWD}/searxng:/etc/searxng" \
		-e "BASE_URL=http://localhost:$PORT/" \
		-e "INSTANCE_NAME=my-instance" \
		searxng/searxng

.PHONY: prepare_for_tests
