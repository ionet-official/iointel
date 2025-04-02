prepare_for_tests:
	export PORT=8080
	docker pull searxng/searxng
	docker stop iointel-searxng || true
	docker run --rm -d \
		-p ${PORT}:8080 \
		-v "${PWD}/searxng:/etc/searxng" \
		-e "BASE_URL=http://localhost:$PORT/" \
		-e "INSTANCE_NAME=my-instance" \
		--name iointel-searxng \
		searxng/searxng
	sleep 5
	docker stop iointel-searxng || true

	yq -i '.search.formats += ["json"]' searxng/settings.yml

	docker run --rm -d \
		-p ${PORT}:8080 \
		-v "${PWD}/searxng:/etc/searxng" \
		-e "BASE_URL=http://localhost:$PORT/" \
		-e "INSTANCE_NAME=my-instance" \
		--name iointel-searxng \
		searxng/searxng


.PHONY: prepare_for_tests
