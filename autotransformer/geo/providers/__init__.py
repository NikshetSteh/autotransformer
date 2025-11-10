from enum import Enum

from .centers import CentersProvider
from .city_boundary import CityBoundaryProvider
from .districts import DistrictsProvider
from .industrial import IndustrialProvider
from .intersections import IntersectionsProvider
from .metro import MetroProvider
from .transport import TransportProvider


class GeoProviderEnum(Enum):
    CENTERS = "centers"
    METRO = "metro"
    TRANSPORT = "transport"
    INTERSECTIONS = "intersections"
    INDUSTRIAL = "industrial"
    CITY_BOUNDARY = "city_boundary"
    DISTRICTS = "districts"


_PROVIDER_REGISTRY = {
    GeoProviderEnum.CENTERS: CentersProvider,
    GeoProviderEnum.METRO: MetroProvider,
    GeoProviderEnum.TRANSPORT: TransportProvider,
    GeoProviderEnum.INTERSECTIONS: IntersectionsProvider,
    GeoProviderEnum.INDUSTRIAL: IndustrialProvider,
    GeoProviderEnum.CITY_BOUNDARY: CityBoundaryProvider,
    GeoProviderEnum.DISTRICTS: DistrictsProvider,
}


def get_provider(name: GeoProviderEnum) -> type:
    return _PROVIDER_REGISTRY[name]
