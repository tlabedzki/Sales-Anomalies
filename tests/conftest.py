import pytest

from sales_anomalies import detect, facts
from tests.shop_factory import make_shop


@pytest.fixture(scope="session")
def shop():
    return make_shop()


@pytest.fixture(scope="session")
def shop_facts(shop):
    return facts.build(shop)


@pytest.fixture(scope="session")
def detection(shop_facts):
    return detect.detect(shop_facts, thresholds={})
