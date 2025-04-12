# check that SUFFIX is set, otherwise give an error
if [ -z "$SUFFIX" ]; then
    echo "SUFFIX is not set. Please set it before running this script. Stopping."
    return 1
fi

# if webhost is not set, raise an error
if [ -z "$WEBHOST" ]; then
    echo "WEBHOST is not set. Please set it before running this script. Stopping."
    return 1
fi


export WA_HOMEPAGE="https://wa-homepage-${SUFFIX}.${WEBHOST}"
export WA_SHOPPING="https://wa-shopping-${SUFFIX}.${WEBHOST}/"
export WA_SHOPPING_ADMIN="https://wa-shopping-admin-${SUFFIX}.${WEBHOST}/admin"
export WA_REDDIT="https://wa-forum-${SUFFIX}.${WEBHOST}"
export WA_GITLAB="https://wa-gitlab-${SUFFIX}.${WEBHOST}"
export WA_WIKIPEDIA="https://wa-wikipedia-${SUFFIX}.${WEBHOST}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="https://wa-openstreetmap-${SUFFIX}.${WEBHOST}"
export WA_FULL_RESET="https://wa-reset-${SUFFIX}.${WEBHOST}"

# visualwebarena

export VWA_HOMEPAGE="https://vwa-homepage-${SUFFIX}.${WEBHOST}"
export VWA_SHOPPING="https://vwa-shopping-${SUFFIX}.${WEBHOST}"
export VWA_REDDIT="https://vwa-forum-${SUFFIX}.${WEBHOST}"
export VWA_CLASSIFIEDS="https://vwa-classifieds-${SUFFIX}.${WEBHOST}"
export VWA_WIKIPEDIA="https://vwa-wikipedia-${SUFFIX}.${WEBHOST}"
export VWA_FULL_RESET="https://vwa-reset-${SUFFIX}.${WEBHOST}"

export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
