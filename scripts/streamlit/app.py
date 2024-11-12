# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import re

# Get the current session
session = get_active_session()

# Page config
st.set_page_config(
    page_title="Connected Mobility - Vehicle Product Quality RCA",
    page_icon=":car:",
    layout="wide",
)

##########################################################################
# Important params
##########################################################################
# Define the coordinates and labels
locations = [
    {
        "label": "Connected Mobility Data - AWS US West 2",
        "latitude": 44.0582,
        "longitude": -121.3153,
        "color": [255, 0, 0],
    },  # Red
    {
        "label": "Manufacturing Data - GCP US Central 1",
        "latitude": 41.8780,
        "longitude": -93.0977,
        "color": [0, 0, 255],
    },  # Blue
    {
        "label": "Supplier Quality Data - Azure US West 2",
        "latitude": 47.7511,
        "longitude": -120.7401,
        "color": [21, 130, 55],
    },  # Green
]


##########################################################################
# Important functions/queries
##########################################################################
# RGB to colour name
def rgb_to_color_name(rgb):
    color_map = {(255, 0, 0): "red", (21, 130, 55): "green", (0, 0, 255): "blue"}
    return color_map.get(tuple(rgb), "unknown")


#######################################
# Connected Mobility
#######################################
def query_vehicles():
    return f"""
    SELECT * FROM connected_mobility.public.vehicles
    """


def query_vehicles_all():
    return f"""
    SELECT 
            car_id, model_year, vehicle_config, doors, 
            state, city, country, 
            -- longitude, latitude,
            longitude::FLOAT AS longitude, latitude::FLOAT AS latitude, 
            date_values, avg_temp_f, avg_wind_speed_mph, tot_precipitation_in, tot_snowfall_in, 
            dtc_error_code
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    """


# Getting a pivot of errors
def query_dtc_errors_counts(year_start, year_end):
    return f"""
    SELECT 
    vzddwd.*,
    dbec.error_code,
    dbec.description
    FROM (
        SELECT dtc_error_code, count(dtc_error_code) AS error_code_count
            FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
            WHERE dtc_error_code <> 0
            AND model_year BETWEEN {year_start} AND {year_end}
            GROUP BY dtc_error_code
            ) AS vzddwd
    JOIN connected_mobility.public.dtc_battery_error_codes AS dbec
    ON vzddwd.dtc_error_code = dbec.error_id
    """


# Model Year vs Error Counts
def query_modelyear_dtc_errors_counts():
    return """
    SELECT model_year, count(dtc_error_code) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    GROUP BY model_year
    ORDER BY model_year
    """


# Doors vs Error Counts
def query_doors_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT  doors.doors, 
            COUNT(dtc.dtc_error_code) AS error_code_count
    FROM 
    (
    SELECT DISTINCT doors 
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    ) AS doors
    LEFT JOIN connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc dtc
        ON doors.doors = dtc.doors 
        AND dtc.dtc_error_code <> 0
        AND dtc.model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY doors.doors
    ORDER BY doors.doors
    """


# Vehicle Config vs Error Counts
def query_vehicleconfig_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT  configs.vehicle_config AS vehicle_config, 
            COUNT(dtc.dtc_error_code) AS error_code_count
    FROM (
    SELECT DISTINCT vehicle_config 
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    ) AS configs
    LEFT JOIN connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc dtc
        ON configs.vehicle_config = dtc.vehicle_config 
        AND dtc.dtc_error_code <> 0
        AND dtc.model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY configs.vehicle_config
    ORDER BY configs.vehicle_config
    """


# State vs Error Counts
def query_state_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT state, count(dtc_error_code) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY state
    ORDER BY state
    """


# Months vs Error Counts
def query_months_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT MONTH(date_values) AS months, COUNT(car_id) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP by months
    ORDER BY months
    """


# Temperature vs Error Counts
def query_temp_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT avg_temp_f, COUNT(car_id) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY avg_temp_f
    ORDER BY avg_temp_f
    """


# Snowfall vs Error Counts
def query_snowfall_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT tot_snowfall_in, COUNT(car_id) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY tot_snowfall_in
    ORDER BY tot_snowfall_in 
    """


# Precipitation vs Error Counts
def query_precipitation_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT tot_precipitation_in, COUNT(car_id) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY tot_precipitation_in
    ORDER BY tot_precipitation_in 
    """


# Battery Part Number vs Error Counts
def query_batterypart_dtc_errors_counts(min_model_year, max_model_year):
    return f"""
    SELECT part_number, COUNT(car_id) AS error_code_count
    FROM connected_mobility.public.vehicles_zipcodes_distances_dates_weather_dtc
    WHERE dtc_error_code <> 0
    AND model_year BETWEEN {min_model_year} AND {max_model_year}
    GROUP BY part_number
    ORDER BY part_number 
    """


#######################################
# Manufacturing
#######################################


# Battery Part Numbers List
def query_battery_part_numbers_list():
    return """
    SELECT DISTINCT part_number FROM connected_mobility.public.vehicles
    """


# Battery Supplier Info with filter
def query_battery_suppliers(part_number):
    return f"""
    SELECT * 
    FROM connected_mobility.public.battery_supplier 
    WHERE id IN (
        SELECT supplier
        FROM connected_mobility.public.part_battery
        WHERE part_number IN {part_number}
    )
    ORDER BY id
    """


# Battery Types Info with fiter
def query_battery_types(part_number):
    return f"""
    SELECT * 
    FROM connected_mobility.public.battery_type 
    WHERE 1=1
    AND id IN (
    SELECT DISTINCT type
    FROM connected_mobility.public.part_battery
    WHERE 1=1
    AND part_number IN {part_number}
    )
    """


# Battery Components Info with filter
def query_battery_components(part_number):
    return f"""
    SELECT * 
    FROM connected_mobility.public.battery_components 
    WHERE 1=1
    AND battery_type IN (
    SELECT DISTINCT type
    FROM connected_mobility.public.part_battery
    WHERE 1=1
    AND part_number IN {part_number}
    )
    """


#######################################
# Supplier Quality Data
#######################################
# Battery Quality Information
def query_battery_quality_info(part_number):
    return f"""
    SELECT  pb.part_number, bs.name, pb.mfg_year, pb.ah, pb.terminal, pb.size_length_cm, 
            pb.type, pb.temp_range_fahrenheit, pb.voltage_range, pb.recommended_charging_voltage_range,
            pb.recommended_charging_current_range, pb.overcharge_protection, pb.overcurrent_protection,
            pb.discharge_current, pb.cut_off_voltage
    FROM connected_mobility.public.part_battery AS pb
    LEFT JOIN  connected_mobility.public.battery_supplier AS bs
    ON pb.supplier = bs.id
    WHERE 1=1
    AND part_number IN {part_number}
    """


#######################################
# RCA Bot
#######################################

# Text2SQL
database_static_prefix = """
Instruction: You are an Snowflake SQL expert. Given an input question, first create a syntactically correct Oracle SQL query and return the query.
You can order the results to return the most informative data in the database.
Pay attention to use only the column names you can see in the schema description below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use TRUNC(SYSDATE) function to get the current date, if the question involves today.
Add column aliases.


Ensure there are no new lines or inappropriate SQL Keywords in the following json. 
It has to be a valid json object with output JSON format as below:
{
    "Instruction": "",
    "Question": "",
    "SQL Query": "",
    "Explanation": "",
    "Note": "",
}

Remove all other information and return only this VALIDATED JSON object.

Schema Description:

"""


schema_txt = open("schema.txt", "r")
schema_description = f"{schema_txt.read()}\n"

prompt_input = ""


def query_parse_json(json_input, scrape):
    return f"""
    SELECT llm_response:"{scrape}"::STRING AS op FROM (
    SELECT PARSE_JSON('{json_input}') AS llm_response);
    """


def query_gen_rca_cortex(model_name, static_prompt):
    return f"""
    SELECT REPLACE(
    SNOWFLAKE.CORTEX.COMPLETE('{model_name}', '{static_prompt}'
    ), '```', ''
    ) AS query_output;
    """


def query_gen_rca_custom(prompt_input):
    return f"""
    SELECT connected_mobility.public.cmllm_text2sql('''{prompt_input}''')::STRING AS query_output;
    """


# Text2Viz
python_static_prefix = """
Instruction: You are a Python developer that writes code using plotly and streamlit to visualize data.
Your data input is a pandas dataframe that you can access with st.session_state.sess_query_output.
Return only the Python code and nothing else.

"""


# Function that extracts the actual Python code returned by mistral
def extract_python_code(text):
    # Regular expression pattern to match Python code after "python" keyword
    pattern = r"(?s)(?<=python).*"

    # Extract Python code using regex
    match = re.search(pattern, text)
    if match:
        python_code = match.group(0).strip()
        print(python_code)
    else:
        print("No Python code found.")


##########################################################################


# Load & cache data
@st.cache_data
def load_data(query_of_interest):
    return session.sql(query_of_interest).to_pandas()


##########################################################################


def set_page(page: str):
    st.session_state.page = page


class Page(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def print_page(self):
        pass

    @abstractmethod
    def print_sidebar(self):
        pass


class BasePage(Page):
    def __init__(self):
        pass

    # Repeatable element: sidebar buttons that navigate to linked pages
    def print_sidebar(self):
        st.title("Vehicle Product Quality - Root Cause Analysis")
        st.divider()
        with st.sidebar:
            st.header("Select Page")
            st.button(
                "Home Page", key="sidebar button 1", on_click=set_page, args=("one",)
            )
            st.button(
                "Connected Mobility Data",
                key="sidebar button 2",
                on_click=set_page,
                args=("two",),
            )
            st.button(
                "Manufacturing Data",
                key="sidebar button 3",
                on_click=set_page,
                args=("three",),
            )
            st.button(
                "Supplier Quality Data",
                key="sidebar button 4",
                on_click=set_page,
                args=("four",),
            )
            st.button(
                "RCA Bot",
                key="sidebar button 5",
                on_click=set_page,
                args=("five",),
            )

    def print_page(self):
        pass


##########################################################################
##########################################################################


class page_one(BasePage):
    def __init__(self):
        self.name = "one"

    def print_sidebar(self):
        super().print_sidebar()

    def print_page(self):
        st.header("Home Page")

        # Add legend using native Streamlit text formatting
        st.write(":red[Connected Mobility Data - AWS US West 2]")
        st.write(":blue[Manufacturing Data - GCP US Central 1]")
        st.write(":green[Supplier Quality Data - Azure US West 2]")
        st.write("3rd party Weather data - Snowflake Marketplace")

        # Define the pairs of coordinates for the arcs
        arcs = [
            {
                "sourcePosition": [loc["longitude"], loc["latitude"]],
                "targetPosition": [locations[0]["longitude"], locations[0]["latitude"]],
                "color": loc["color"],
            }
            for loc in locations[1:]
        ]

        # Create a Pydeck ScatterplotLayer with customized text properties
        layers = []
        for location in locations:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=[location],
                get_position=["longitude", "latitude"],
                get_radius=100000,
                get_fill_color="color",
                pickable=True,
            )
            layers.append(layer)

        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arcs,
            getSourcePosition="sourcePosition",
            getTargetPosition="targetPosition",
            getSourceColor="color",
            getTargetColor="color",
            getWidth=5,
        )
        layers.append(arc_layer)

        view_state = pdk.ViewState(
            latitude=42, longitude=-97, zoom=3, bearing=-10, pitch=45
        )

        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/streets-v11",
            tooltip={"text": "{label}"},
        )

        # Show the Pydeck chart
        st.pydeck_chart(r)

        st.divider()
        st.divider()
        st.divider()


##########################################################################
##########################################################################


class page_two(BasePage):
    def __init__(self):
        self.name = "two"

    def print_sidebar(self):
        super().print_sidebar()

    def print_page(self):
        st.header("Connected Mobility Data")

        # Loading data
        vehicles = load_data(query_vehicles())
        vehicles_all = load_data(query_vehicles_all())

        # Filter out rows with missing data
        vehicles_all_subset = vehicles_all[
            ["LATITUDE", "LONGITUDE", "VEHICLE_CONFIG", "MODEL_YEAR", "DTC_ERROR_CODE"]
        ].dropna()
        min_year = int(vehicles_all["MODEL_YEAR"].min())
        max_year = int(vehicles_all["MODEL_YEAR"].max())

        # Formatting Page
        col1, col2 = st.columns(2)
        with col1:
            # Add a slider for model year filtering
            selected_year = st.slider(
                "Select Model Year", min_year, max_year, (min_year, max_year)
            )

            # Filter data based on selected year range
            filtered_vehicles = vehicles_all_subset[
                (vehicles_all_subset["MODEL_YEAR"] >= selected_year[0])
                & (vehicles_all_subset["MODEL_YEAR"] <= selected_year[1])
            ]

            # Display filtered data
            with st.expander("Filtered Data:"):
                st.write(filtered_vehicles.head())
                st.caption(
                    "Filtered Data Counts: "
                    + str(filtered_vehicles["MODEL_YEAR"].count())
                )

            # Create the bar chart
            dtc_error_counts = load_data(
                query_dtc_errors_counts(selected_year[0], selected_year[1])
            )
            fig = go.Figure(
                [
                    go.Bar(
                        x=dtc_error_counts["ERROR_CODE"],
                        y=dtc_error_counts["ERROR_CODE_COUNT"],
                    )
                ]
            )

            # Add titles and labels
            fig.update_layout(
                title="DTC Error Code Counts",
                xaxis_title="DTC Error Codes",
                yaxis_title="DTC Error Code Counts",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            st.write("**Legend:**")
            st.caption("1. P1794 - Battery Voltage Circuit Malfunction")
            st.caption("2. B1317 - Battery Voltage High")
            st.caption("3. B1318 - Battery Voltage Low")
            st.caption("4. B1671 - Battery Module Voltage Out Of Range")
            st.caption("5. B1676 - Battery Pack Voltage Out Of Range")
            st.caption(
                "6. C2100 - Battery Voltage Low - Circuit Voltage Below Threshold"
            )

        modelyear_dtc_errors_counts = load_data(query_modelyear_dtc_errors_counts())
        doors_dtc_errors_counts = load_data(
            query_doors_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        vehicleconfig_dtc_errors_counts = load_data(
            query_vehicleconfig_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        state_dtc_errors_counts = load_data(
            query_state_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        months_dtc_errors_counts = load_data(
            query_months_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        temp_dtc_errors_counts = load_data(
            query_temp_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        snowfall_dtc_errors_counts = load_data(
            query_snowfall_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        precipitation_dtc_errors_counts = load_data(
            query_precipitation_dtc_errors_counts(selected_year[0], selected_year[1])
        )
        batterypart_dtc_errors_counts = load_data(
            query_batterypart_dtc_errors_counts(selected_year[0], selected_year[1])
        )

        with col2:
            # Metrics
            filtered_veh = vehicles[
                (vehicles["MODEL_YEAR"] >= selected_year[0])
                & (vehicles["MODEL_YEAR"] <= selected_year[1])
            ]
            vehicles_count = filtered_veh["CAR_ID"].count()
            sedan_count = len(
                filtered_veh.loc[filtered_veh["VEHICLE_CONFIG"] == "sedan"]
            )
            suv_count = vehicles_count - sedan_count
            svs_count = f"{sedan_count} | {suv_count}"
            door2_count = len(filtered_veh.loc[filtered_veh["DOORS"] == 2])
            door4_count = vehicles_count - door2_count
            d2v4_count = f"{door2_count} | {door4_count}"

            sub1, sub2, sub3 = st.columns(3)
            with sub1:
                # st.metric('Country', 'USA')
                st.metric("Number of vehicles", vehicles_count)
            with sub2:
                st.metric("Sedans vs SUV", svs_count)
            with sub3:
                st.metric("2 door vs 4 door", d2v4_count)

            tab1, tab2, tab3 = st.tabs(
                [
                    "Model Year",
                    "Sedans vs SUV",
                    "2 door vs 4 door",
                ]
            )
            with tab1:
                # Filter data based on selected year range
                modelyear_dtc_errors_counts = modelyear_dtc_errors_counts[
                    (modelyear_dtc_errors_counts["MODEL_YEAR"] >= selected_year[0])
                    & (modelyear_dtc_errors_counts["MODEL_YEAR"] <= selected_year[1])
                ]

                # Model Year vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=modelyear_dtc_errors_counts["MODEL_YEAR"],
                            y=modelyear_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Model Year vs DTC Error Code Counts",
                    xaxis_title="Model Year",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Vehicle Config vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=vehicleconfig_dtc_errors_counts["VEHICLE_CONFIG"],
                            y=vehicleconfig_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Vehicle Config vs DTC Error Code Counts",
                    xaxis_title="Vehicle Config",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                # Doors vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=doors_dtc_errors_counts["DOORS"],
                            y=doors_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Doors vs DTC Error Code Counts",
                    xaxis_title="Doors",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)

            # State vs DTC Error Code Counts
            fig = go.Figure(
                [
                    go.Bar(
                        x=state_dtc_errors_counts["STATE"],
                        y=state_dtc_errors_counts["ERROR_CODE_COUNT"],
                    )
                ]
            )

            # Add titles and labels
            fig.update_layout(
                title="State vs DTC Error Code Counts",
                xaxis_title="State",
                yaxis_title="DTC Error Code Counts",
            )
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            # Months vs DTC Error Code Counts
            fig = go.Figure(
                [
                    go.Bar(
                        x=months_dtc_errors_counts["MONTHS"],
                        y=months_dtc_errors_counts["ERROR_CODE_COUNT"],
                    )
                ]
            )

            # Add titles and labels
            fig.update_layout(
                title="Month of the year vs DTC Error Code Counts",
                xaxis_title="Month of the year",
                yaxis_title="DTC Error Code Counts",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            tab1, tab2, tab3 = st.tabs(["Temperature", "Snowfall", "Precipitation"])
            with tab1:
                # Average Temperature (f) vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=temp_dtc_errors_counts["AVG_TEMP_F"],
                            y=temp_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Average Temperature (f) vs DTC Error Code Counts",
                    xaxis_title="Average Temperature (f)",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                # Total Snowfall (in) vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=snowfall_dtc_errors_counts["TOT_SNOWFALL_IN"],
                            y=snowfall_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Total Snowfall (in) vs DTC Error Code Counts",
                    xaxis_title="Total Snowfall (in)",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab3:
                # Total Precipitation (in) vs DTC Error Code Counts
                fig = go.Figure(
                    [
                        go.Bar(
                            x=precipitation_dtc_errors_counts["TOT_PRECIPITATION_IN"],
                            y=precipitation_dtc_errors_counts["ERROR_CODE_COUNT"],
                        )
                    ]
                )

                # Add titles and labels
                fig.update_layout(
                    title="Total Precipitation (in) vs DTC Error Code Counts",
                    xaxis_title="Total Precipitation (in)",
                    yaxis_title="DTC Error Code Counts",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Battery Part Number vs DTC Error Code Counts
        fig = go.Figure(
            [
                go.Bar(
                    x=batterypart_dtc_errors_counts["PART_NUMBER"],
                    y=batterypart_dtc_errors_counts["ERROR_CODE_COUNT"],
                )
            ]
        )

        # Add titles and labels
        fig.update_layout(
            title="Battery Part Number vs DTC Error Code Counts",
            xaxis_title="Battery Part Number",
            yaxis_title="DTC Error Code Counts",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.divider()
        st.divider()


##########################################################################
##########################################################################


class page_three(BasePage):
    def __init__(self):
        self.name = "three"

    def print_sidebar(self):
        super().print_sidebar()

    def print_page(self):
        st.header("Manufacturing Data")

        # Loading data
        battery_part_numbers_list = load_data(query_battery_part_numbers_list())
        # Filtering options
        options_list = [["88AHH362021"], ["72AHT312020"]]
        options = st.multiselect(
            "Choose the battery part numbers of interest",
            options=battery_part_numbers_list.values.tolist(),
            default=options_list,
        )

        options_flat = [item for sublist in options for item in sublist]
        options_str = tuple(options_flat)

        battery_suppliers = load_data(query_battery_suppliers(options_str))
        battery_types = load_data(query_battery_types(options_str))
        battery_components = load_data(query_battery_components(options_str))

        battery_suppliers.rename(
            columns={"STATE": "MANUFACTURING_FACILITY"}, inplace=True
        )

        # Formatting Page
        col1, col2 = st.columns(2)
        with col1:
            st.caption(
                "List of Suppliers and their Manufacturing Facilities that supplied the defective batteries:"
            )
            st.write(battery_suppliers, use_container_width=True)
            st.caption("The Battery types affected:")
            st.write(battery_types, use_container_width=True)
            st.caption(
                "The components that went into making these defective batteries based on type:"
            )
            st.write(battery_components, use_container_width=True)
        with col2:
            # Create the Map
            view_state = pdk.ViewState(
                latitude=42,
                longitude=-97,
                zoom=2.5,
            )

            # Convert DataFrame to list of dictionaries
            data_list = battery_suppliers.to_dict(orient="records")

            layers = []
            for data in data_list:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[data],
                    get_position=["LONGITUDE", "LATITUDE"],
                    get_radius=100000,
                    pickable=True,
                )
                layers.append(layer)

            map_chart = pdk.Deck(
                map_style="mapbox://styles/mapbox/streets-v11",
                initial_view_state=view_state,
                layers=layers,
                tooltip={"text": "Battery Mfg: {NAME}, {MANUFACTURING_FACILITY}"},
            )
            st.pydeck_chart(map_chart, use_container_width=True)

        st.divider()
        st.divider()
        st.divider()


##########################################################################
##########################################################################


class page_four(BasePage):
    def __init__(self):
        self.name = "four"

    def print_sidebar(self):
        super().print_sidebar()

    def print_page(self):
        st.header("Supplier Quality Data")

        # Loading data
        battery_part_numbers_list = load_data(query_battery_part_numbers_list())

        # Filtering options
        options_list = [["88AHH362021"], ["72AHT312020"]]
        options = st.multiselect(
            "Choose the battery part numbers of interest",
            options=battery_part_numbers_list.values.tolist(),
            default=options_list,
        )
        # All options
        all_options_list = battery_part_numbers_list.values.tolist()

        options_flat = [item for sublist in options for item in sublist]
        options_str = tuple(options_flat)
        # All options
        all_options_flat = [item for sublist in all_options_list for item in sublist]
        all_options_str = tuple(all_options_flat)

        battery_quality_info = load_data(query_battery_quality_info(options_str))
        st.write(battery_quality_info)

        # All options
        all_battery_quality_info = load_data(
            query_battery_quality_info(all_options_str)
        )

        # Add a column to battery_quality_info
        all_battery_quality_info["QUALITY_METRIC"] = [
            random.randint(-1, 3) for _ in range(len(all_battery_quality_info))
        ]
        condition = all_battery_quality_info["PART_NUMBER"].isin(options_flat)
        all_battery_quality_info.loc[condition, "QUALITY_METRIC"] = [
            random.randint(-3, -2) for _ in range(condition.sum())
        ]

        st.subheader("Temperature vs Battery Performance")

        # Adding some context
        st.caption(
            """This bell curve shows the Battery part's performance metrics vs temperature during production.
        While all batteries are of acceptable/approved quality, some maybe of *low performance* as per defined thresholds.
        But with our ongoing similarity and differentiation analysis, we need to amend and exercise tighter quality controls 
        to prevent such failures in batteries from recurring."""
        )

        # Creating a bell curve
        quality_metric = all_battery_quality_info["QUALITY_METRIC"]
        part_numbers = all_battery_quality_info["PART_NUMBER"]
        sup_names = all_battery_quality_info["NAME"]

        mu = quality_metric.mean()
        sigma = quality_metric.std()
        x = np.linspace(quality_metric.min(), quality_metric.max(), 100)
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode="lines"))

        fig = go.Figure(
            data=go.Scatter(x=x, y=y, mode="lines", name="Bell Curve", showlegend=False)
        )

        # Define regions for different quality levels
        low_quality_x = np.linspace(-3, -1, 100)
        low_quality_y = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((low_quality_x - mu) ** 2) / (2 * sigma**2))
        )
        medium_quality_x = np.linspace(-1, 1, 100)
        medium_quality_y = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((medium_quality_x - mu) ** 2) / (2 * sigma**2))
        )
        high_quality_x = np.linspace(1, 3, 100)
        high_quality_y = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((high_quality_x - mu) ** 2) / (2 * sigma**2))
        )

        # Add vertical fills
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([low_quality_x, low_quality_x[::-1]]),
                y=np.concatenate([low_quality_y, np.zeros_like(low_quality_y)]),
                fill="toself",
                fillcolor="rgba(255,100,100,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Low Performers",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([medium_quality_x, medium_quality_x[::-1]]),
                y=np.concatenate([medium_quality_y, np.zeros_like(medium_quality_y)]),
                fill="toself",
                fillcolor="rgba(255,255,100,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Average Performers",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([high_quality_x, high_quality_x[::-1]]),
                y=np.concatenate([high_quality_y, np.zeros_like(high_quality_y)]),
                fill="toself",
                fillcolor="rgba(100,255,100,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="High Performers",
                showlegend=False,
            )
        )

        # Add annotations
        fig.add_annotation(
            text="Low Performers", x=-2, y=0.08, font=dict(size=12, color="red")
        )
        fig.add_annotation(
            text="Average Performers", x=0, y=0.08, font=dict(size=12, color="orange")
        )
        fig.add_annotation(
            text="High Performers", x=2, y=0.08, font=dict(size=12, color="green")
        )

        for part_number, quality_met, sup_name in zip(
            part_numbers, quality_metric, sup_names
        ):
            fig.add_trace(
                go.Scatter(
                    x=[quality_met],
                    y=[0],
                    mode="markers",
                    marker=dict(size=8),
                    name=part_number,
                    text=[f"Supplier Name: {sup_name}<br>Part Number: {part_number}"],
                    hoverinfo="text",
                )
            )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.divider()
        st.divider()


##########################################################################
##########################################################################


class page_five(BasePage):
    def __init__(self):
        self.name = "five"

    def print_sidebar(self):
        super().print_sidebar()

    def print_page(self):
        st.header("RCA Bot")

        # Formatting Page
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Write a prompt in plain & simple English to the RCA Bot:")

            prompt_input = "\nQuestion:\n" + st.text_area(
                label="", placeholder="Start writing here..."
            )
            static_prompt = database_static_prefix + schema_description + prompt_input
            # st.write(static_prompt)

            if "sess_prompt_output" not in st.session_state:
                st.session_state.sess_prompt_output = ""

            if "sess_query_output" not in st.session_state:
                st.session_state.sess_query_output = ""

            selectbox_categories = {
                "Cortex (Recommended)": [
                    "snowflake-arctic",
                    "mistral-large",
                    "reka-flash",
                    "mixtral-8x7b",
                    "llama2-70b-chat",
                    "mistral-7b",
                    "gemma-7b",
                ],
                "Custom": [
                    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M"
                ],
            }
            selectbox_parent_category = ""
            col3, col4 = st.columns(2)
            with col3:
                selectbox_parent_category = st.selectbox(
                    label="Choose Model Category",
                    options=["Cortex (Recommended)", "Custom"],
                )
            with col4:
                selectbox_category = st.selectbox(
                    label="Choose a Model",
                    options=selectbox_categories[selectbox_parent_category],
                )

            button_gen_sql_query = st.button("Generate SQL Query")
            if button_gen_sql_query:
                if (
                    selectbox_category
                    == "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M"
                ):
                    st.session_state.sess_prompt_output = (
                        load_data(query_gen_rca_custom(prompt_input))["QUERY_OUTPUT"][0]
                        .split("Answer")[0]
                        .replace("\\", "")
                        .replace("sql", "")
                        .replace(
                            """Sure, here is the SQL query for the given question:\n\n""",
                            "",
                        )
                        .replace(
                            """Sure, here is the SQL query for the question:\n\n""", ""
                        )
                        .replace(
                            """Sure, here is the SQL query for the question:\n\nsql""",
                            "",
                        )
                        .strip()
                    )
                else:
                    st.session_state.sess_prompt_output = (
                        load_data(
                            query_gen_rca_cortex(selectbox_category, static_prompt)
                        )["QUERY_OUTPUT"][0]
                        .replace("json", "")
                        .replace("  ", "")
                        .replace("\n", " ")
                        .replace("\\n", " ")
                        .replace("'", "''")
                        .replace("\\_", "_")
                        .strip()
                    )
                    # st.write(st.session_state.sess_prompt_output)
                    try:
                        st.session_state.sess_prompt_output = load_data(
                            query_parse_json(
                                st.session_state.sess_prompt_output, "SQL Query"
                            )
                        )["OP"][0]
                    except:
                        st.session_state.sess_prompt_output = (
                            "Parsing issue with the model, try another model please."
                        )

                    # st.session_state.sess_prompt_output = load_data(query_gen_rca_cortex(selectbox_category, static_prompt))["QUERY_OUTPUT"][0].\
                    # split("Answer")[0].\
                    # replace('\\', '').\
                    # replace('sql', '').\
                    # replace('''Sure, here is the SQL query for the given question:\n\n''', '').\
                    # replace('''Sure, here is the SQL query for the question:\n\n''', '').\
                    # replace('''Sure, here is the SQL query for the question:\n\nsql''', '').\
                    # strip()

        with col2:
            st.caption("RCA Bot response:")

            st.text_area(
                label="",
                value=st.session_state.sess_prompt_output,
                placeholder="Generated SQL output here",
            )
            col5, col6 = st.columns(2)
            with col5:
                exec_query = st.button("Execute SQL Query")
            with col6:
                plot_type_categories = [
                    "Histogram",
                    "Bar Chart",
                    "Line Chart",
                    "Scatter Plot",
                    "Pie Chart",
                ]
                plot_category = st.selectbox(
                    label="Choose a Plot Type",
                    options=plot_type_categories,
                )
                exec_viz = st.button("Plot it!")

        st.divider()

        if exec_query:
            try:
                st.session_state.sess_query_output = load_data(
                    f"{st.session_state.sess_prompt_output}"
                )
            except:
                st.error(
                    """SQL Errors encountered, please change your prompt or try a different model. 
                If after repeated attemps you are still seeing this message, please contact your administrator."""
                )
            else:
                st.subheader("Query Results:")
                st.write(st.session_state.sess_query_output)

        if exec_viz:
            col7, col8 = st.columns(2)

            col_list = st.session_state.sess_query_output.columns
            col_list = ", ".join(st.session_state.sess_query_output.columns)
            col_list = ", ".join([f'"{item.strip()}"' for item in col_list.split(",")])
            # st.write(col_list)

            col_list_prompt = f"""Use the following columns from the pandas dataframe st.session_state.sess_query_output:\n{col_list}."""
            plot_category_prompt = (
                "\n"
                + "\n"
                + f"""Create an interactive {plot_category} based on the above information."""
            )
            python_prompt = (
                python_static_prefix + col_list_prompt + plot_category_prompt
            )
            # st.write(python_prompt)

            with col7:
                st.subheader("Query Results:")
                st.write(st.session_state.sess_query_output)
            with col8:
                try:
                    st.session_state.sess_python_output = load_data(
                        query_gen_rca_cortex(selectbox_category, python_prompt)
                    )["QUERY_OUTPUT"][0]
                    # st.write(st.session_state.sess_python_output)
                except:
                    st.error(
                        """SQL Errors encountered, please change your prompt or try a different model. 
                        If after repeated attemps you are still seeing this message, please contact your administrator."""
                    )
                else:
                    python_code = st.session_state.sess_python_output.replace(
                        "python", ""
                    )
                    # st.code(python_code, language="python", line_numbers=False)

                    st.subheader("Plot:")
                    try:
                        exec(python_code)
                    except:
                        st.error(
                            """Python Errors encountered, please change your prompt or try a different model. 
                            If after repeated attemps you are still seeing this message, please contact your administrator."""
                        )
                    else:
                        del st.session_state.sess_prompt_output
                        del st.session_state.sess_query_output
                        del st.session_state.sess_python_output


##########################################################################
##########################################################################

if "session" not in st.session_state:
    st.session_state.session = get_active_session()

if "page" not in st.session_state:
    st.session_state.page = "one"

pages = [page_one(), page_two(), page_three(), page_four(), page_five()]


def main():
    for page in pages:
        if page.name == st.session_state.page:
            page.print_sidebar()
            page.print_page()


main()