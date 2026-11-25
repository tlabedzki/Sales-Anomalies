# Sales Anomalies

[![Tests](https://github.com/tlabedzki/Sales-Anomalies/actions/workflows/tests.yml/badge.svg)](https://github.com/tlabedzki/Sales-Anomalies/actions/workflows/tests.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Studium przypadku: codzienna kontrola sprzedaży fikcyjnego sklepu Lumo, która znajduje nietypowe dni, rozkłada je na
przyczyny w złotych i opisuje je zdaniem dla kierownika.

**Problem.** Kierownik sklepu internetowego widzi rano sprzedaż z wczoraj, ale nie wie, czy spadek o kilkanaście
procent to zwykły szum, czy sygnał, że coś się zepsuło: rabat, przewoźnik, kampania albo płatności na telefonach.
Zanim ktoś to sprawdzi ręcznie, mijają dni, a pieniądze uciekają.

**Podejście.** Kod codziennie porównuje każdy fragment sklepu z prognozą liczoną tylko z przeszłości, rozkłada
każde odchylenie na przyczyny, które sumują się co do grosza, a model językowy opisuje gotowe liczby i niczego nie
liczy sam. Wszystko jest oceniane kluczem odpowiedzi: anomaliami wstrzykniętymi do danych o znanej przyczynie
i zdarzeniami, które oznaczył sam generator danych.

<!-- generated:result -->
**Wynik.** W dwóch latach dziennych danych wykrywanie zgłosiło 27 anomalii; 15 z nich to zdarzenia z klucza odpowiedzi
(56%), przy 0,50 fałszywego alarmu na miesiąc. Znalazło 9 z 31 wstrzykniętych scenariuszy, każdy rozkład zgadza się ze
zmianą co do grosza, a główna przyczyna jest trafna w 15 z 15 prawdziwych anomalii.
<!-- /generated:result -->

<!-- generated:hero -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/app-demo-dark.svg">
  <img alt="Aplikacja Sales Anomalies w ilustracji: oś czasu z 27 anomaliami z lat 2023–2024; kliknięcie kropki Odzież (15–24 lipca 2023) otwiera notatkę dla kierownika i rozbicie zmiany +157 553 zł na przyczyny, z główną: zwroty (+157 553 zł); potem powrót na oś czasu." src="docs/img/app-demo.svg">
</picture>
<!-- /generated:hero -->

**Stack:** Python · pandas · NumPy · DuckDB · Plotly · Dash · Claude API · pytest

<!-- generated:summary -->
- **Trafione:** płatność nie działa na telefonach (wszystkie przypadki); w większości: fala zwrotów wadliwej partii (3 z
  4), wyższa prowizja afiliacyjna (2 z 3); awarie sklepu z `anomaly_data` (2 z 2). Każde z 27 rozbić sumuje się do
  zmiany co do grosza.
- **Chybione:** spada ruch z wyszukiwarki, konkurent tnie ceny, kampania przestaje działać, dodatkowy rabat w kategorii
  i kanale, zbyt niska cena hitów (0 trafień): ich zmiany mają medianę 2,9 odchylenia dziennego szumu i giną w nim. Z
  267 oszustw z `anomaly_data` wykrywanie znalazło 0.
<!-- /generated:summary -->

## 1. Sklep Lumo i pytania kierownika

Lumo to fikcyjny sklep z [Data-Generatora](https://github.com/tlabedzki/Data-Generator): 15 kategorii, 6 kanałów
ruchu, ok. 730 linii zamówień dziennie w latach 2021–2024. Kierownik co rano chce wiedzieć, czy wczoraj albo
w ostatnim tygodniu coś odbiegło od normy, w której części sklepu, dlaczego i ile to kosztowało.

| Pytanie | Jak odpowiada analiza | Gdzie leży prawdziwa odpowiedź |
|---|---|---|
| Czy w którejś części sklepu sprzedaż, marża, zwroty, koszty albo konwersja odbiegły od normy? | Prognoza z przeszłości i z-score na dzień i na 7 dni (rozdział 3) | Scenariusze wstrzyknięte przez projekt i `anomaly_data` |
| Które zmiany wykrywanie łapie, a które gubi? | Ocena każdej anomalii kluczem odpowiedzi (rozdział 4) | `scenario_truth`: rodzaj, segment, dni i wielkość każdej zmiany |
| Dlaczego wynik się zmienił i za ile złotych? | Most marży i rozkład wolumenu LMDI (rozdział 5) | Przyczyna wpisana w scenariusz albo w zdarzenie generatora |
| Jak powiedzieć to kierownikowi bez zmyślonych liczb? | Notatka modelu językowego sprawdzana kodem (rozdział 6) | Arkusz faktów policzony przez kod |

## 2. Skąd wiadomo, co jest prawdą

Dane są w całości syntetyczne, więc dla każdej zmiany wiadomo, czy naprawdę zaszła i dlaczego. Projekt korzysta
z dwóch niezależnych kluczy odpowiedzi.

- **Scenariusze wstrzyknięte przez projekt** (`scenarios.py`). Na danych generatora kod dopisuje co dwa tygodnie jedno
  zdarzenie jednego z dziewięciu rodzajów, np. dodatkowy rabat w jednej kategorii i jednym kanale, falę zwrotów wadliwej
  partii, wyższą stawkę przewoźnika, konkurenta, który tnie ceny, kampanię, która przestaje działać, albo płatność,
  która nie działa na telefonach. Każde zdarzenie zmienia tabele tak, jak zrobiłoby to prawdziwe: niższa cena obniża też
  płatność, prowizję i późniejszy zwrot, a utracone zamówienie zabiera ze sobą przesyłkę i płatność. Wpływ w złotych to
  różnica danych po wstrzyknięciu i przed nim, więc jest dokładny.
- **Zdarzenia oznaczone przez generator** (tabela `anomaly_data`): oszustwa (przejęte konto składa w 2–3 dni kilka
  drogich zamówień), całodniowe awarie sklepu i fala zwrotów jednego produktu.

Analiza nie widzi żadnego z tych kluczy. Anomalia liczy się jako prawdziwa tylko wtedy, gdy zdarzenie dotyczy jej
dni i segmentu oraz wyjaśnia co najmniej 30% odchylenia, które zobaczyła. Przy kilkuset oszustwach któreś zawsze
gdzieś trwa, a luźniejsze dopasowanie przypisywałoby im każdy alarm w pobliżu.

Dwie kontrole powinny wyjść bliskie zera. Pierwsza to okres bez wstrzyknięć (marzec–maj 2024): każdy alarm, którego
nie wyjaśnia generator, jest tam fałszywy. Druga to ten sam sklep w ogóle bez scenariuszy. Progi alarmów
dobrano na drugim sklepie z innym ziarnem losowania (seed 7), a wyniki podaje sklep z ziarnem 42. Nic nie jest więc
strojone na danych, na których jest oceniane.

## 3. Czy coś się dziś stało?

**Pytanie.** Czy któraś część sklepu odbiegła wczoraj albo w ostatnim tygodniu od tego, czego można się było
spodziewać?

**Co robi analiza.** Dla każdego ze 112 segmentów (cały sklep, 6 kanałów, 15 kategorii i 90 par kategoria × kanał)
model prognozuje każdy dzień tylko z dwóch lat przed nim. To regresja Poissona (model dla liczb zdarzeń i kwot) z dniem
tygodnia, miesiącem, trendem, dniami specjalnymi sklepu i kalendarzem kampanii, przeliczana co miesiąc, jak w algorytmie
Farringtona do nadzoru nad zachorowaniami. Do tego dochodzi poprawka o poziom z ostatnich tygodni, bo ceny i asortyment
powoli się zmieniają. Odchylenie mierzy z-score (o ile odchyleń standardowych dzień jest od prognozy) na jeden dzień i
na 7 dni. Alarm wymaga z-score ponad progiem swojej grupy i zmiany ponad próg istotności w złotych. Jedno drogie
zamówienie nie jest anomalią: do testu skoku liczy się jako typowe, a do testu spadku pojedyncze linie są przycinane
do 99. percentyla ich kategorii.

**Jak to sprawdzamy.** Precyzja to odsetek alarmów, które wyjaśnia klucz odpowiedzi (1 = żadnego fałszywego
alarmu). Czułość to odsetek zdarzeń z klucza, które wykrywanie znalazło.

**Wynik.**

<!-- generated:detection -->
![27 anomalii w 2023–2024: 15 prawdziwych, 12 fałszywych alarmów](docs/img/timeline.png)

Okres bez wstrzyknięć (3 miesiące): 2 fałszywe alarmy. Ten sam sklep bez żadnego scenariusza: 16 anomalii, w tym 13
fałszywych alarmów (0,54 na miesiąc).
<!-- /generated:detection -->

**Co to znaczy dla kierownika.** Przy takim progu alarm jest rzadki i częściej prawdziwy niż fałszywy, więc warto go
sprawdzić tego samego dnia. Próg to decyzja biznesowa: niższy znalazłby więcej, ale zabrałby więcej czasu na
fałszywe tropy.

## 4. Co wykrywanie łapie, a co gubi?

**Pytanie.** Które rodzaje problemów kontrola widzi od razu, a które przechodzą niezauważone?

**Co robi analiza.** Każde wstrzyknięte zdarzenie i każde zdarzenie generatora dostaje ocenę: znalezione albo nie.
Siła sygnału scenariusza to jego prawdziwa zmiana w złotych (albo w zamówieniach) podzielona przez szum, jakiego model
spodziewa się w tym segmencie i tych dniach; mówi, ile odchyleń standardowych dzieli zmianę od zwykłego dnia.

**Jak to sprawdzamy.** Zdarzenie jest znalezione, gdy co najmniej jedna anomalia dotyczy jego dni i segmentu oraz
wyjaśnia ją w co najmniej 30%.

**Wynik.**

<!-- generated:recall -->
![W całości wykryte: płatność nie działa na telefonach, awaria sklepu na cały dzień](docs/img/recall.png)

![Najsilniejszy przeoczony sygnał miał 4,5 odchylenia szumu, najsłabszy wykryty 6,2](docs/img/sizes.png)

Wykryte scenariusze mają medianę sygnału 14,0 odchylenia szumu, przeoczone 2,9; najsłabszy wykryty 6,2, najsilniejszy
przeoczony 4,5.
<!-- /generated:recall -->

**Co to znaczy dla kierownika.** Kontrola dzienna łapie awarie i duże przecieki. Dzienna sprzedaż Lumo odbiega
jednak od prognozy wyraźnie nawet bez żadnego problemu, więc kilkunastoprocentowy spadek w jednym kanale albo
kategorii przez parę dni zlewa się z tym szumem. Takie zmiany trzeba śledzić innymi miarami: rabat średni na linię,
koszt paczki, odsetek zwrotów.

## 5. Dlaczego? Przyczyny w złotych co do grosza

**Pytanie.** Skąd wzięła się zmiana i ile złotych przypada na każdą przyczynę?

**Co robi analiza.** Most marży, czyli rozpisanie różnicy między wartością oczekiwaną a rzeczywistą na składniki,
tak jak w controllingu: wolumen, ceny i asortyment, rabat, wysyłka, opłaty za płatności, prowizja, a gdy jedno konto
z kilkoma zamówieniami robi dużą część zmiany, także zamówienia jednego konta. Wolumen dzieli się dalej metodą LMDI
(logarytmiczny indeks Divisii, który rozkłada iloczyn na składniki bez reszty) na odwiedziny, konwersję i udział
segmentu w zamówieniach. Zwroty dzielą się według powodu zwrotu, koszty według rodzaju, a konwersja według
urządzenia. Nazwę przyczyny wolumenu kod nadaje na podstawie dowodów: brak ruchu w całym sklepie to awaria, kanał
z trwającą kampanią to kampania, a kategoria, w której konkurent obniżył ceny, to ceny konkurencji. Części są
zaokrąglane metodą największych reszt, więc ich suma równa się zmianie co do grosza.

**Jak to sprawdzamy.** Suma części musi się równać zmianie (test na każdej anomalii), a główna przyczyna musi być
przyczyną zdarzenia z klucza odpowiedzi.

**Wynik.**

<!-- generated:bridge -->
![Główna przyczyna: zwroty, +157 553 zł, czyli 100% zmiany](docs/img/bridge.png)

Notatka do tej anomalii (z szablonu kodu): „15–24 lipca 2023: koszt zwrotów (Odzież) +157 553 zł (+133%) względem
oczekiwań, głównie przez: zwroty (+157 553 zł); warto sprawdzić partię produktu z największą liczbą zwrotów.”
<!-- /generated:bridge -->

**Co to znaczy dla kierownika.** Zamiast „marża spadła” dostaje konkretną przyczynę z kwotą i wie, gdzie zacząć:
w ustawieniach rabatów, u przewoźnika albo w kampanii.

## 6. Zdanie dla kierownika: kod liczy, AI tłumaczy

**Pytanie.** Jak zamienić rozkład na jedno albo dwa zdania, które nie przekręcą liczb?

**Co robi analiza.** Kod zapisuje fakty anomalii w arkuszu gotowych napisów: segment, dni, zmianę w złotych
i procentach, trzy główne przyczyny z kwotami, dowody i podpowiedź, co sprawdzić. Model Claude Haiku 5.5 (najtańszy
aktualny model Claude, z wyjściem w ustalonym formacie JSON) przepisuje arkusz na notatkę. Kod sprawdza każdą liczbę
w notatce: musi stać w arkuszu, inaczej notatka wraca do modelu z listą błędów, a po drugiej odmowie zostaje
zdanie z szablonu kodu. Notatki są zapisane w repozytorium (`data/explanations.json`), więc aplikacja i README
działają bez klucza API.

**Jak to sprawdzamy.** Evale w `evals/` oceniają kodem pierwszą wersję notatki z każdego przypadku: zero liczb spoza
arkusza, najwyżej dwa zdania i 50 słów, nazwana główna przyczyna, segment i kwota, zgodny kierunek zmiany.

**Wynik.**

<!-- generated:notes -->
0 z 54 notatek (po polsku i po angielsku) napisał claude-haiku-5-5; 0 przeszło dopiero za drugim razem, po odrzuceniu
pierwszej wersji przez kontrolę liczb, a pozostałe mają zdanie z szablonu kodu. W notatkach, które trafiły do aplikacji,
nie ma ani jednej liczby spoza arkusza faktów. Wszystkie wywołania kosztowały 0,0000 USD (0 tokenów wejścia, 0 wyjścia).
<!-- /generated:notes -->

**Co to znaczy dla kierownika.** Czyta zdanie jak od analityka, a każda liczba w nim pochodzi z kodu.

## 7. Analiza kontra prawda

<!-- generated:verdicts -->
Wykrywanie zgłosiło 27 anomalii, z czego 12 to fałszywe alarmy. Kontrole: okres bez wstrzyknięć 2 fałszywe alarmy, sklep
bez scenariuszy 0,54 fałszywego alarmu na miesiąc.

| Zdarzenie | Źródło | Znalezione |
|---|---|---|
| Płatność nie działa na telefonach | wstrzyknięte | 3 z 3 |
| Fala zwrotów wadliwej partii | wstrzyknięte | 3 z 4 |
| Wyższa prowizja afiliacyjna | wstrzyknięte | 2 z 3 |
| Przewoźnik podnosi stawkę | wstrzyknięte | 1 z 4 |
| Spada ruch z wyszukiwarki | wstrzyknięte | 0 z 3 |
| Konkurent tnie ceny | wstrzyknięte | 0 z 3 |
| Kampania przestaje działać | wstrzyknięte | 0 z 3 |
| Dodatkowy rabat w kategorii i kanale | wstrzyknięte | 0 z 4 |
| Zbyt niska cena hitów | wstrzyknięte | 0 z 4 |
| Awaria sklepu na cały dzień | oznaczone przez generator | 2 z 2 |
| Wadliwa partia jednego produktu | oznaczone przez generator | 0 z 1 |
| Oszustwo (przejęte konto) | oznaczone przez generator | 0 z 267 |
<!-- /generated:verdicts -->

**Gdzie metody się pomyliły i czego to uczy.**

- **Rabaty i błędne ceny w jednej kategorii.** Rabat o 15–50 punktów procentowych w jednej kategorii i kanale przez
  kilka dni zjada dużą część marży tej komórki, ale komórka ma kilkadziesiąt linii dziennie, a marża pojedynczej
  linii rozrzuca się o kilkaset złotych. Lekcja: rabat warto pilnować wprost (średni rabat na linię), nie przez
  marżę.
- **Kampanie, ruch z wyszukiwarki i konkurencja.** Model wie, kiedy kampania trwa, ale nie zna jej siły, a sprzedaż
  kanału mocno waha się z dnia na dzień. Nawet spadek o połowę przez trzy dni nie przebija się przez ten szum.
  Lekcja: ruch i konwersję kanału trzeba śledzić w sesjach, nie tylko w złotych.
- **Oszustwa z `anomaly_data`.** Przejęte konto składa kilka drogich zamówień w paru komórkach, po jednym czy dwa
  w każdej, a dzienna sprzedaż elektroniki i tak mocno się waha. To anomalia konta, nie segmentu.
  Lekcja: oszustwa łapie się na poziomie klienta (wiele drogich zamówień z jednego konta), nie na sumach sprzedaży.
- **Fałszywe alarmy.** Większość to skoki sprzedaży w kanale mediów społecznościowych w czasie kampanii (model nie zna
  jej siły) i w listopadowym szczycie: 11.11 i Black Friday. To realne zmiany, tylko nie ma ich w kluczu odpowiedzi.

Pomyłki pokazane wprost to część wyniku. Klucz odpowiedzi umie powiedzieć „przeoczone” i „fałszywy alarm”, więc
„znalezione” coś znaczy, a granica wykrywalności jest zmierzona, nie zgadnięta.

## 8. Wnioski i ograniczenia

**Rekomendacje.**

1. Codzienna kontrola na sumach sprzedaży i marży nadaje się do wykrywania awarii, przecieków kosztów i dużych fal
   zwrotów; warto ją uruchamiać każdego ranka.
2. Rabaty, ceny i kampanie w pojedynczych segmentach wymagają osobnych, prostszych miar (średni rabat, cena
   najlepiej sprzedających się produktów, sesje kanału), bo w sumach giną w szumie.
3. Każdy alarm warto dostarczać razem z rozkładem na przyczyny: kwota przy przyczynie mówi, od czego zacząć.

**Ograniczenia.** Dane są syntetyczne, a szum dzienny wynika z budowy generatora; w prawdziwym sklepie dochodzą
pogoda, awarie integracji i sezon, a klucza odpowiedzi nie ma. Wtedy sprawdzianem zostaje okres bez znanych
zdarzeń i ręczna weryfikacja części alarmów. Ocena dotyczy jednego sklepu (seed 42), a progi dobrano na jednym innym
(seed 7). Scenariusze nie nakładają się na siebie i omijają listopad i grudzień. Próg istotności w złotych i budżet
fałszywych alarmów to decyzje przyjęte w projekcie.

## Szybki start

Potrzebujesz Gita i Pythona 3.11. Sklonuj repozytorium:

```bash
git clone https://github.com/tlabedzki/Sales-Anomalies.git
cd Sales-Anomalies
```

<details><summary>Utwórz i włącz środowisko wirtualne z Pythonem 3.11 (Windows, macOS i Linux)</summary>

Podanie wersji wybiera Pythona 3.11, nawet jeśli domyślny jest inny.

| Krok | Windows (PowerShell) | macOS i Linux |
|---|---|---|
| Utworzenie | `py -3.11 -m venv .venv` | `python3.11 -m venv .venv` |
| Włączenie | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Sprawdzenie | `python --version` pokazuje `Python 3.11.x` | `python --version` pokazuje `Python 3.11.x` |

Jeśli PowerShell odpowie, że uruchamianie skryptów jest wyłączone, wykonaj raz
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` i włącz środowisko ponownie.

</details>

Potem zainstaluj pakiety, wygeneruj dane, uruchom kontrolę i aplikację:

```bash
pip install -r requirements.txt
python scripts/get_data.py
python main.py
python -m ui.app
```

`get_data.py` klonuje Data-Generator v1.7.1 i generuje dwa sklepy (ok. 3 minuty). `main.py` liczy dwa lata
kontroli, ocenę kluczem i kontrolę bez scenariuszy (ok. 4 minut) i zapisuje wyniki do `output/`. Aplikacja działa pod
adresem http://127.0.0.1:8050; zatrzymasz ją przez Ctrl+C.

### Model językowy

Notatki są zapisane w repozytorium, więc klucz nie jest potrzebny. Żeby napisać brakujące notatki na żywo, wpisz
`ANTHROPIC_API_KEY=...` do pliku `.env` w katalogu projektu i uruchom `python main.py --live`. Progi alarmów są w
repozytorium; `python scripts/calibrate.py` dobiera je od nowa na sklepie z ziarnem 7. Evale: `python -m evals.run`
(jeden przebieg) albo `python -m evals.run --repeat 3`.

## Dane

Dane są syntetyczne. [Data-Generator](https://github.com/tlabedzki/Data-Generator) v1.7.1 (licencja MIT) generuje
fikcyjny sklep; `scripts/get_data.py` uruchamia go bez zmian w kodzie, z nadpisanymi ustawieniami: lata 2021–2024,
1 060 000 linii zamówień, 400 000 klientów, ziarno 42 (wyniki) i 7 (progi). To ok. 10 razy więcej niż domyślnie,
tyle co średni sklep internetowy. Nazwy marek w danych są prawdziwe, produkty fikcyjne.

```mermaid
erDiagram
    product_data ||--o{ order_data : "product_id"
    order_data ||--o| shipment_data : "order_id"
    order_data ||--o| payment_data : "order_id"
    order_data ||--o{ return_data : "order_line_id"
    session_data ||--o| order_data : "order_id"
    campaign_data ||--o{ session_data : "campaign_id"
    product_data ||--o{ competitor_prices : "product_id"
    product_data {
        int product_id PK
        string category
    }
    order_data {
        int order_line_id PK
        int order_id
        date order_date
        string channel
    }
    shipment_data {
        int order_id FK
        float carrier_cost
    }
    payment_data {
        int order_id FK
        float payment_fee
    }
    return_data {
        int order_line_id FK
        date return_date
    }
    session_data {
        int session_id PK
        int order_id FK
    }
    campaign_data {
        int campaign_id PK
        date start_date
    }
    competitor_prices {
        int product_id FK
        float price_gross
    }
```

Klucz odpowiedzi to tabela `anomaly_data` generatora (bez relacji kluczami: zdarzenie wskazuje dni, tabelę i konto)
oraz `scenario_truth` tego projektu (`output/events.csv`). Kwoty są netto, bez 23% VAT.

## Struktura projektu

<details><summary>Pokaż drzewo</summary>

```
main.py                        kontrola dwóch lat, ocena kluczem, zapis do output/
scripts/get_data.py            Data-Generator v1.7.1 -> data/raw/seed-42, seed-7
scripts/calibrate.py           progi alarmów z seed 7 -> src/sales_anomalies/thresholds.json
scripts/make_charts.py         wykresy README i bloki generated (readme_blocks.py)
scripts/make_demo_svg.py       animowany hero z zrzutów aplikacji
src/sales_anomalies/
  economics.py                 ekonomia linii i zwrotów (SQL w DuckDB)
  facts.py                     dzienne fakty 112 segmentów
  baseline.py                  prognoza z przeszłości (regresja Poissona, poziom)
  detect.py                    z-score, alarmy, epizody, najwęższy segment
  calibrate.py                 progi z seed 7
  scenarios.py                 wstrzyknięte anomalie i ich prawda
  truth.py, evaluate.py        klucz odpowiedzi i ocena
  decompose.py                 przyczyny w złotych co do grosza
  explain.py                   notatki modelu i kontrola liczb
evals/                         przypadki, oceny kodem, wyniki
ui/app.py                      aplikacja Dash + Mantine
tests/                         pytest na małym sklepie syntetycznym
```

</details>

## Testy

```bash
pip install -r requirements-dev.txt
python -m pytest
```

48 testów sprawdza na małym sklepie syntetycznym (bez plików danych i bez sieci, ok. 15 s): ekonomię linii, prognozę
tylko z przeszłości, wstrzykiwanie scenariuszy, wykrywanie, sumę części co do grosza, kontrolę liczb w notatkach
i widoki aplikacji.

## Licencja

MIT, zobacz [LICENSE](LICENSE). [Changelog](CHANGELOG.md).

Ostatnia aktualizacja README: 2026-11-25
