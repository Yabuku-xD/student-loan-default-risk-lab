# Watchlist Summary

## Random Forest
- Recommended threshold: `0.29` (Precision 8.80%, Recall 78.57%)
- Institutions flagged: **901** | Avg default 0.677 | Avg net price $11,067 | Avg completion 67.44%

Top flagged institutions:

| INSTNM                                              | control_label      | region_label    | dominant_program   |   default_rate_pct |   risk_score |
|:----------------------------------------------------|:-------------------|:----------------|:-------------------|-------------------:|-------------:|
| Manuel and Theresa's School of Hair Design          | Private for-profit | Southwest       | Trades & Applied   |                2.8 |        0.847 |
| Lawrenceburg Technical College                      | Private for-profit | Southeast       | Trades & Applied   |               14.2 |        0.843 |
| Piedmont Community College                          | Public             | Southeast       | Humanities         |               14.2 |        0.834 |
| LeGrand Institute of Cosmetology Inc                | Private for-profit | Southeast       | Trades & Applied   |                2.7 |        0.832 |
| Academy of Cosmetology Inc                          | Private for-profit | Rocky Mountains | Trades & Applied   |                2.4 |        0.816 |
| Halifax Community College                           | Public             | Southeast       | Health             |                7.6 |        0.815 |
| Cosmetology School of Arts & Sciences               | Private for-profit | Rocky Mountains | Trades & Applied   |                2.8 |        0.812 |
| Lawrence & Company College of Cosmetology           | Private for-profit | Far West        | Trades & Applied   |                3.9 |        0.803 |
| Manuel and Theresa's School of Hair Design-Victoria | Private for-profit | Southwest       | Trades & Applied   |                2.8 |        0.799 |
| Dolce The Academy                                   | Private for-profit | New England     | Trades & Applied   |                2.4 |        0.799 |

## Balanced Random Forest
- Recommended threshold: `0.68` (Precision 11.36%, Recall 71.43%)
- Institutions flagged: **605** | Avg default 0.998 | Avg net price $10,642 | Avg completion 67.82%

Top flagged institutions:

| INSTNM                                       | control_label      | region_label   | dominant_program                |   default_rate_pct |   risk_score |
|:---------------------------------------------|:-------------------|:---------------|:--------------------------------|-------------------:|-------------:|
| Alaska Christian College                     | Private nonprofit  | Far West       | Humanities                      |                2.3 |        0.999 |
| Grace Mission University                     | Private nonprofit  | Far West       | Social Science & Public Service |                9   |        0.999 |
| Nuvani Institute                             | Private for-profit | Southwest      | Trades & Applied                |                2.1 |        0.998 |
| Southeastern Community College               | Public             | Southeast      | Humanities                      |                7.1 |        0.998 |
| James Sprunt Community College               | Public             | Southeast      | Humanities                      |               16.6 |        0.998 |
| Greene County Career and Technology Center   | Public             | Mid East       | Health                          |                2.1 |        0.998 |
| Shawsheen Valley School of Practical Nursing | Public             | New England    | Health                          |                2.2 |        0.997 |
| Welder Training and Testing Institute        | Private for-profit | Mid East       | Trades & Applied                |                2.6 |        0.997 |
| North Georgia Technical College              | Public             | Southeast      | Trades & Applied                |                4.1 |        0.997 |
| Alliance Computing Solutions                 | Private for-profit | Mid East       | Health                          |                4.2 |        0.997 |

## Smote Logistic
- Recommended threshold: `0.67` (Precision 12.50%, Recall 25.00%)
- Institutions flagged: **207** | Avg default 0.648 | Avg net price $9,529 | Avg completion 71.11%

Top flagged institutions:

| INSTNM                                  | control_label      | region_label   | dominant_program                |   default_rate_pct |   risk_score |
|:----------------------------------------|:-------------------|:---------------|:--------------------------------|-------------------:|-------------:|
| Southwestern Christian College          | Private nonprofit  | Southwest      | Humanities                      |                4.5 |        0.986 |
| Nouvelle Institute                      | Private for-profit | Southeast      | Trades & Applied                |                0   |        0.982 |
| Nuvani Institute                        | Private for-profit | Southwest      | Trades & Applied                |                2.1 |        0.977 |
| Rosedale Bible College                  | Private nonprofit  | Great Lakes    | Social Science & Public Service |                0   |        0.968 |
| Metropolitan Learning Institute         | Private nonprofit  | Mid East       | Health                          |                0   |        0.965 |
| Grace International Beauty School       | Private for-profit | Mid East       | Trades & Applied                |                0   |        0.964 |
| Word of Life Bible Institute            | Private nonprofit  | Mid East       | Social Science & Public Service |                0   |        0.963 |
| Jamestown Business College              | Private for-profit | Mid East       | Business                        |                3.7 |        0.961 |
| Manhattan School of Computer Technology | Private nonprofit  | Mid East       | Humanities                      |                0   |        0.96  |
| CET-Soledad                             | Private nonprofit  | Far West       | Trades & Applied                |                0   |        0.96  |
