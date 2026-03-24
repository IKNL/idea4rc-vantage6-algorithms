VANTAGE6_VERSION ?= 5.0.0
TAG ?= tbd
REGISTRY ?= harbor2.vantage6.ai
REGISTRY_PROJECT ?= idea4rc
PLATFORMS ?= linux/amd64
BASE ?= 5.0

# We use a conditional (true on any non-empty string) later. To avoid
# accidents, we don't use user-controlled PUSH_REG directly.
# See: https://www.gnu.org/software/make/manual/html_node/Conditional-Functions.html
PUSH_REG ?= false
_condition_push :=
ifeq ($(PUSH_REG), true)
	_condition_push := not_empty_so_true
endif

help:
	@echo "Usage:"
	@echo "  make help                - show this message"
	@echo "  make image-sessions      - build sessions image"
	@echo "  make image-analytics     - build analytics image"
	@echo "  make image-preprocessing - build preprocessing image"
	@echo "  make images              - build all images"
	@echo ""
	@echo "Using "
	@echo "  registry:  ${REGISTRY}/${REGISTRY_PROJECT}"
	@echo "  tag:       ${TAG}-v6-${VANTAGE6_VERSION}"
	@echo "  base:      ${BASE}"
	@echo "  platforms: ${PLATFORMS}"
	@echo "  vantage6:  ${VANTAGE6_VERSION}"
	@echo ""

define build_image
	@echo "Building ${REGISTRY}/${REGISTRY_PROJECT}/$(1):${TAG}-v6-${VANTAGE6_VERSION}"
	@echo "Building ${REGISTRY}/${REGISTRY_PROJECT}/$(1):${TAG}"
	docker buildx build \
		--tag ${REGISTRY}/${REGISTRY_PROJECT}/$(1):${TAG}-v6-${VANTAGE6_VERSION} \
		--tag ${REGISTRY}/${REGISTRY_PROJECT}/$(1):${TAG} \
		--platform ${PLATFORMS} \
		--build-arg TAG=${TAG} \
		--build-arg BASE=${BASE} \
		-f ./docker/$(2).Dockerfile \
		$(if ${_condition_push},--push .,.)
endef

image-sessions:
	$(call build_image,sessions,sessions)

image-analytics:
	$(call build_image,analytics,analytics)

image-preprocessing:
	$(call build_image,preprocessing,preprocessing)

images: image-sessions image-analytics image-preprocessing
