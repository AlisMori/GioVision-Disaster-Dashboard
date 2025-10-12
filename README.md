# GeoVision Disaster Dashboard (GDD)

### ICT305 – Data Visualisation and Simulation | Murdoch University, 2025

---

## Overview

**GeoVision Disaster Dashboard (GDD)** is an ongoing group project developed as part of Murdoch University's *ICT305 Data Visualisation and Simulation* unit. The project aims to create an interactive web-based dashboard that visualizes global natural disaster data, integrating both real-time and historical information.

The goal is to design a tool that helps users — such as policymakers, NGOs, researchers, and the public — explore and understand how disasters impact different regions, populations, and economies worldwide.

---

## Project Objectives

* Combine multiple open-source disaster datasets into a unified visualization platform.
* Deliver near real-time global insights through APIs (NASA EONET, GDACS).
* Provide historical trend analysis using EM-DAT data.
* Build an intuitive, interactive dashboard for exploring disaster frequency, severity, and impact.

---

## Data Sources

| Dataset        | Source                                                                       | Type     | Description                                                                |
| -------------- | ---------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------- |
| **NASA EONET** | [NASA Earth Observatory Natural Event Tracker](https://eonet.gsfc.nasa.gov/) | JSON API | Real-time data on natural events such as wildfires, storms, and volcanoes. |
| **GDACS**      | [Global Disaster Alert and Coordination System](https://www.gdacs.org/)      | XML RSS  | Real-time disaster alerts with severity levels and estimated impacts.      |
| **EM-DAT**     | [CRED International Disaster Database](https://www.emdat.be/)                | CSV      | Historical data on disaster events, human impact, and economic losses.     |

Each dataset provides complementary insights — EONET and GDACS offer near real-time monitoring, while EM-DAT adds validated long-term records for historical analysis.

---

## Planned Features

* **Interactive map:** Global view of current and historical disasters.
* **Trend analysis:** Visualization of disaster frequency and type over time.
* **Live alerts:** Real-time updates on ongoing global disasters.
* **Impact comparison:** Country-level statistics on affected populations and economic losses.
* **Filtering tools:** Explore data by region, disaster type, and time range.

---

## Development Tools & Technologies

| Category                 | Tools               |
| ------------------------ | ------------------- |
| **Programming Language** | Python 3.10+        |
| **Framework**            | Streamlit (planned) |
| **Data Processing**      | Pandas, Requests    |
| **Visualization**        | Plotly, Folium      |
| **Data Formats**         | JSON, XML, CSV      |

> This list will be refined as the project progresses and toolset decisions are finalized.

---

## Project Status

This project is **currently in active development**. The data pipeline, dashboard structure, and visual design are still evolving.
The repository will be updated regularly as components are implemented, tested, and refined.

---

## Installation & Usage (Work in Progress)

When complete, the dashboard will be installable and runnable locally via Python. Tentative setup steps:

```bash
git clone https://github.com/AlisMori/GioVision-Disaster-Dashboard.git
cd GioVision-Disaster-Dashboard
python -m venv venv
source venv/bin/activate # (Linux/Mac)
venv\Scripts\activate # (Windows)
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Final setup instructions will be added upon project completion.

---

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## Academic Context

This project is developed as part of the coursework for **ICT305 Data Visualisation and Simulation** at **Murdoch University (2025)**.
It is for educational purposes only, using publicly available open datasets from NASA, GDACS, and CRED.

---
