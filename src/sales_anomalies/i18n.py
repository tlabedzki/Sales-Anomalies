"""Labels and number formats in Polish and English (README and descriptions in the repo's language, app in English)."""
from __future__ import annotations

import pandas as pd

LANGS = ("pl", "en")
MINUS = "−"

CATEGORIES_PL = {
    "Automotive": "Motoryzacja", "Baby & Kids": "Dziecko", "Beauty": "Uroda", "Books & Media": "Książki i multimedia",
    "Clothing": "Odzież", "DIY Tools": "Narzędzia", "Electronics": "Elektronika", "Furniture": "Meble",
    "Health": "Zdrowie", "Home & Garden": "Dom i ogród", "Jewelry & Accessories": "Biżuteria i dodatki",
    "Office": "Biuro", "Pet Supplies": "Zwierzęta", "Sports": "Sport", "Toys": "Zabawki",
}
CHANNELS = {
    "pl": {"affiliate": "afiliacja", "direct": "wejścia bezpośrednie", "email": "e-mail",
           "organic_search": "wyszukiwarka", "paid_search": "reklamy w wyszukiwarce",
           "social": "media społecznościowe"},
    "en": {"affiliate": "affiliate", "direct": "direct", "email": "email", "organic_search": "organic search",
           "paid_search": "paid search", "social": "social media"},
}
METRICS = {
    "pl": {"sales": "sprzedaż", "margin": "marża", "returns": "koszt zwrotów", "costs": "koszty obsługi zamówień",
           "conversion": "konwersja"},
    "en": {"sales": "sales", "margin": "margin", "returns": "cost of returns", "costs": "order handling costs",
           "conversion": "conversion"},
}
DIRECTIONS = {"pl": {"up": "wzrost", "down": "spadek"}, "en": {"up": "rise", "down": "drop"}}
CAUSES = {
    "pl": {"discount": "rabaty", "prices": "ceny i asortyment", "returns": "zwroty", "shipping": "koszty wysyłki",
           "payment_fees": "opłaty za płatności", "commission": "prowizja afiliacyjna",
           "competitor_prices": "ceny konkurencji", "campaign": "kampania", "conversion": "konwersja",
           "traffic": "ruch na stronie", "demand": "popyt", "outage": "awaria sklepu",
           "single_account": "zamówienia jednego konta"},
    "en": {"discount": "discounts", "prices": "prices and product mix", "returns": "returns",
           "shipping": "shipping costs", "payment_fees": "payment fees", "commission": "affiliate commission",
           "competitor_prices": "competitor prices", "campaign": "a campaign", "conversion": "conversion",
           "traffic": "site traffic", "demand": "demand", "outage": "a shop outage",
           "single_account": "orders of one account"},
}
SUGGESTIONS = {
    "pl": {"discount": "sprawdzić ustawienia rabatów i kody promocyjne",
           "prices": "sprawdzić cennik najlepiej sprzedających się produktów",
           "returns": "sprawdzić partię produktu z największą liczbą zwrotów",
           "shipping": "sprawdzić stawki przewoźnika", "payment_fees": "sprawdzić stawki operatora płatności",
           "commission": "sprawdzić rozliczenie z siecią afiliacyjną",
           "competitor_prices": "porównać ceny z konkurentem",
           "campaign": "sprawdzić, czy kampania wyświetla reklamy",
           "conversion": "sprawdzić koszyk i płatności na telefonach",
           "traffic": "sprawdzić pozycje w wyszukiwarce i źródła ruchu",
           "demand": "obserwować kategorię w kolejnych dniach", "outage": "sprawdzić dostępność sklepu i logi",
           "single_account": "zweryfikować zamówienia tego konta przed wysyłką"},
    "en": {"discount": "check discount settings and voucher codes",
           "prices": "check the price list of the best sellers",
           "returns": "check the batch of the most returned product", "shipping": "check the carrier's rates",
           "payment_fees": "check the payment operator's rates",
           "commission": "check the affiliate network's invoice",
           "competitor_prices": "compare prices with the competitor",
           "campaign": "check that the campaign is still serving ads",
           "conversion": "check the cart and payments on phones",
           "traffic": "check search rankings and traffic sources",
           "demand": "watch the category over the next days", "outage": "check the shop's uptime and logs",
           "single_account": "verify this account's orders before shipping"},
}
REASONS_PL = {"wrong_size": "zły rozmiar", "changed_mind": "zmiana zdania", "not_as_described": "niezgodny z opisem",
              "defective": "wadliwy", "damaged": "uszkodzony", "wrong_item": "pomyłka w zamówieniu"}
DEVICES_PL = {"mobile": "telefony", "desktop": "komputery", "tablet": "tablety"}
MONTHS = {
    "pl": ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
           "października", "listopada", "grudnia"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"],
}


def category(name: str, lang: str) -> str:
    return CATEGORIES_PL.get(name, name) if lang == "pl" else name


def channel(name: str, lang: str) -> str:
    return CHANNELS[lang].get(name, name)


def segment(key: str, lang: str) -> str:
    cat, chan = key.split("|")
    if cat == "*" and chan == "*":
        return "cały sklep" if lang == "pl" else "whole shop"
    if cat == "*":
        return f"kanał {channel(chan, lang)}" if lang == "pl" else f"{channel(chan, lang)} channel"
    if chan == "*":
        return category(cat, lang)
    return f"{category(cat, lang)} · {channel(chan, lang)}"


def number(value: float, lang: str, decimals: int = 0) -> str:
    text = f"{abs(value):,.{decimals}f}"
    if lang == "pl":
        text = text.replace(",", " ").replace(".", ",")
    return (MINUS if value < 0 and round(abs(value), decimals) > 0 else "") + text


def signed(value: float, lang: str, decimals: int = 0) -> str:
    body = number(value, lang, decimals)
    return body if body.startswith(MINUS) or round(value, decimals) == 0 else f"+{body}"


def money(value: float, lang: str, sign: bool = False) -> str:
    body = signed(value, lang) if sign else number(value, lang)
    if lang == "pl":
        return f"{body} zł"
    if body.startswith(("+", MINUS)):
        return f"{body[0]}PLN {body[1:]}"
    return f"PLN {body}"


def percent(value: float, lang: str, sign: bool = True, decimals: int = 0) -> str:
    return (signed(value * 100, lang, decimals) if sign else number(value * 100, lang, decimals)) + "%"


def day(value: pd.Timestamp, lang: str, year: bool = True) -> str:
    text = f"{value.day} {MONTHS[lang][value.month - 1]}"
    return f"{text} {value.year}" if year else text


def period(start: pd.Timestamp, end: pd.Timestamp, lang: str) -> str:
    if start == end:
        return day(start, lang)
    if start.year == end.year and start.month == end.month:
        return f"{start.day}–{day(end, lang)}"
    if start.year == end.year:
        return f"{day(start, lang, year=False)} – {day(end, lang)}"
    return f"{day(start, lang)} – {day(end, lang)}"


def days(count: int, lang: str) -> str:
    if lang == "en":
        return f"{count} day" + ("" if count == 1 else "s")
    return f"{count} " + ("dzień" if count == 1 else "dni")
