NUMERICAL_FEATURES = [
    'year',
    'floor',
    'rooms',
    'total_area',
    'living_area',
    'longitude',
    'latitude',
    'description',
    'image_urls'
]

BOOLEAN_FEATURES = [
    'price_verification',
    'apartment_verification'
]

CATEGORIAL_FEATURES = [
    'heating',
    'walls',
    'region',
    'city',
]

TARGET = 'price'

SUBSTITUTE_FEATURES = [
    'description',
    'image_urls'
]

USELESS_FEATURES = [
    'title',
    'seller',
    'street',
    'publish_date',
    'offer_id',
    'apartment_id'
]

