prepare_for_tests:
	export PORT=8080
	docker pull searxng/searxng
	docker stop iointel-searxng || true

	# Run searxng to generate default settings.yml
	# Then edit settings.yml to support json format in the output, and restart docker with updated settings
	docker run --rm -d \
		-p ${PORT}:8080 \
		-v "${PWD}/searxng:/etc/searxng" \
		-e "BASE_URL=http://localhost:${PORT}/" \
		-e "INSTANCE_NAME=my-instance" \
		--name iointel-searxng \
		searxng/searxng
	sleep 5
	docker stop iointel-searxng || true

	sudo python3 -c "import yaml; f='searxng/settings.yml'; d=yaml.safe_load(open(f)); d.setdefault('search',{}).setdefault('formats',[]).append('json'); open(f,'w').write(yaml.safe_dump(d))"

	docker run --rm -d \
		-p ${PORT}:8080 \
		-v "${PWD}/searxng:/etc/searxng" \
		-e "BASE_URL=http://localhost:${PORT}/" \
		-e "INSTANCE_NAME=my-instance" \
		--name iointel-searxng \
		searxng/searxng


.PHONY: prepare_for_tests
