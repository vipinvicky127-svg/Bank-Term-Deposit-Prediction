# Bank Term Deposit Dataset - Data Dictionary

This document details the schema, definitions, expected data types, and domain significance for all features in the Bank Marketing dataset.

---

## 1. Client Demographic and Financial Information

| Variable Name | Type | Description | Values / Examples |
|:---|:---|:---|:---|
| `ID` | Integer | Unique identifier for each client / record. | `26110`, `40576`, ... |
| `age` | Integer | Age of the client in years. | `18` - `95` |
| `job` | Categorical | Type of occupation/profession held by the client. | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| `marital` | Categorical | Marital status of the client. | `married`, `single`, `divorced` (note: `divorced` includes widowed) |
| `education` | Categorical | Highest level of education attained. | `primary`, `secondary`, `tertiary`, `unknown` |
| `default` | Binary / Categorical | Does the client currently have credit in default? | `no`, `yes` |
| `balance` | Integer | Average yearly balance in client account (in Euros). | e.g. `-8019` to `102127` |
| `housing` | Binary / Categorical | Does the client have an active housing loan? | `no`, `yes` |
| `loan` | Binary / Categorical | Does the client have a personal loan? | `no`, `yes` |

---

## 2. Campaign Interaction Information

| Variable Name | Type | Description | Values / Examples |
|:---|:---|:---|:---|
| `contact` | Categorical | Communication channel used for contact. | `cellular`, `telephone`, `unknown` |
| `day` | Integer | Day of the month of the last contact. | `1` - `31` |
| `month` | Categorical | Month of the year of the last contact. | `jan`, `feb`, `mar`, ..., `nov`, `dec` |
| `duration` | Integer | Duration of the last phone call in seconds. | `>= 0` (e.g. `44`, `867`) |
| `campaign` | Integer | Total number of contacts performed during this campaign for this client (including the last contact). | `>= 1` |
| `pdays` | Integer | Number of days since client was last contacted in a previous campaign. | `-1` (never contacted before), or `>= 0` |
| `previous` | Integer | Number of contacts performed before this campaign for this client. | `>= 0` |
| `poutcome` | Categorical | Outcome of the previous marketing campaign. | `failure`, `other`, `success`, `unknown` |

---

## 3. Target Variable

| Variable Name | Type | Description | Values |
|:---|:---|:---|:---|
| `subscribed` | Binary / Categorical | Has the client subscribed to a term deposit? | `yes` (subscribed), `no` (did not subscribe) |

---

## Notes & Modeling Best Practices

1. **`duration`**: Highly predictive of the target (longer calls correlate with interested clients). In real-world operational environments, duration is only known after the call concludes. For prospect ranking prior to placing calls, models can be evaluated with and without call duration.
2. **`ID` Column**: A unique surrogate identifier that must not be used as a predictive feature.
3. **Missing Values**: Features with `unknown` categories represent unrecorded attributes. Rather than dropping rows, treating `unknown` as an explicit categorical level captures informative missingness patterns.
