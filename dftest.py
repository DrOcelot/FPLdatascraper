import os
import requests
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from pandasgui import show as pg_show

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# globals
manager_list = []
player_list = []
league_base_url = 'https://fantasy.premierleague.com/api/'  # the url that starts all API calls
manager_base_url = 'https://fantasy.premierleague.com/api/entry/'  # The url that gets basic manager data
league_id = 71964  # this is the essex classic league
max_retries = 10
manager_path = os.path.join("C:\\Users\\owenp\\Documents\\JSONS", "managers.json")
manager_file_exists = os.path.exists(manager_path)
# end of globals


# general functions
def generate_column_names(num_picks):
    columns = []
    for i in range(1, num_picks + 1):
        columns.extend([
            f'pick_{i}_id',
            f'pick_{i}_web_name',
            f'pick_{i}_points',
            f'pick_{i}_role',
            f'pick_{i}_mult',
            f'pick_{i}_c',
            f'pick_{i}_vc',
            f'pick_{i}_auto_sub',
            f'pick_{i}_dgw'
        ])
    return columns

# end of general functions.
# classes


class Manager:
    def __init__(self, manager_id, name, entered_events, gw_df, picks_df):
        self.manager_id = manager_id
        self.name = name
        self.entered_events = entered_events  # which game weeks this manager played in
        self.gw_df = gw_df  # a dataframe containing information for each game week
        # contains 'points', 'points on bench' and
        # 'picks' which is a list of player ids picked that week
        self.picks_df = picks_df
        self.manager_df = None

    def print_manager(self):  # print manager info
        print(self.manager_id)
        print(self.name)
        print(self.entered_events)
        print(self.gw_df)
        print(self.picks_df)
        print(self.manager_df)

    def bench_sum(self):  # total up the points on the bench through whole season
        return self.gw_df['points_on_bench'].sum()

    def print_picks(self, gw):  # print the list of picks the manager made in a given game week
        picks_gw = self.picks_df.loc[self.picks_df['GW'] == gw, 'picks'].values[0]
        for pick in picks_gw:
            player_name = df_player_information.at[df_player_information['id'] == pick.player_id, 'web_name'].values
            print(player_name)

    def suboptimal_bench_n(self):
        swap_points_df = pd.DataFrame(columns=["manager", "n_gk_swaps", "total_gk_points", "bench_diff"])
        new_row = {'manager': self.name}
        swap_points_df.loc[0] = new_row

        gk_df = pd.DataFrame({'GW': range(1, 39)})  # create a new data frame with 38 game weeks
        gk_df['gk_points'] = None
        gk_df['gk_bench_points'] = None
        gk_df['bench_diff'] = None
        prev_gk_id = None
        n_gk_swaps = 0
        for gw in self.entered_events:
            gk_id = None
            gk_bench_id = None
            picks_gw = self.picks_df.loc[self.picks_df['GW'] == gw, 'picks'].iloc[0]
            for pick in picks_gw:
                if pick.pos == 1:
                    gk_id = pick.player_id
                elif pick.pos == 12:
                    gk_bench_id = pick.player_id

            # Initialize an empty set to track seen game weeks
            seen_gws = set()

            # Loop through the history of the GK
            gk_row_index = df_player_information.index[df_player_information['id'] == gk_id][0]
            for points_h in df_player_information.at[gk_row_index, 'history']:
                # Check if the current game week matches the specified one
                if points_h.gw == gw:
                    # Check if the game week has been seen before
                    if points_h.gw in seen_gws:
                        if isinstance(points_h.points, (int, float)):
                            gk_df.loc[gk_df['GW'] == gw, 'gk_points'] += points_h.points
                        else:
                            raise ValueError("points_h.points should be a scalar (int or float).")
                    else:
                        gk_df.loc[gk_df['GW'] == gw, 'gk_points'] = points_h.points
                        # Add the game week to the set of seen game weeks
                        seen_gws.add(points_h.gw)

            seen_gws = set()
            gk_bench_row_index = df_player_information.index[df_player_information['id'] == gk_bench_id][0]
            for points_h in df_player_information.at[gk_bench_row_index, 'history']:
                # Check if the current game week matches the specified one
                if points_h.gw == gw:
                    # Check if the game week has been seen before
                    if points_h.gw in seen_gws:
                        if isinstance(points_h.points, (int, float)):
                            gk_df.loc[gk_df['GW'] == gw, 'gk_bench_points'] += points_h.points
                        else:
                            raise ValueError("points_h.points should be a scalar (int or float).")

                    else:
                        gk_df.loc[gk_df['GW'] == gw, 'gk_bench_points'] = points_h.points
                        # Add the game week to the set of seen game weeks
                        seen_gws.add(points_h.gw)

            gk_df.loc[gk_df['GW'] == gw, 'bench_diff'] = (
                    gk_df.loc[gk_df['GW'] == gw, 'gk_points'] - gk_df.loc[gk_df['GW'] == gw, 'gk_bench_points'])

            if prev_gk_id != gk_id:
                n_gk_swaps += 1
            prev_gk_id = gk_id

        # condition = gk_df['gk_bench_points'] > gk_df['gk_points']
        # count = condition.sum()
        # print(self.name, " picked the wrong goalkeeper ", count, " times!",
        #      " Their picked goalkeeper scored ", gk_df['gk_points'].sum(), " across the season.")
        # print(self.name, " swapped goalkeeper ", n_gk_swaps, " times!")

        swap_points_df.loc[swap_points_df['manager'] == self.name, 'n_gk_swaps'] = n_gk_swaps
        swap_points_df.loc[swap_points_df['manager'] == self.name, 'total_gk_points'] = gk_df['gk_points'].sum()
        swap_points_df.loc[swap_points_df['manager'] == self.name, 'bench_diff'] = gk_df['bench_diff'].sum()
        return swap_points_df

    def construct_data_frame(self):
        # Initialize DataFrame with default columns
        initial_data = {
            'manager_id': self.manager_id, 'manager_name': self.name, 'GW': range(1, 39), 'gw_entered': pd.NA
        }
        new_column_names = generate_column_names(15)
        for col in new_column_names:
            initial_data[col] = pd.NA
        initial_data['gw_points'] = pd.NA
        initial_data['gw_bench_points'] = pd.NA
        initial_data['chips_used'] = pd.NA

        # Create the DataFrame in a single operation
        self.manager_df = pd.DataFrame(initial_data)

        # Prepare updates in a list
        updates = []

        for gw in self.entered_events:
            update = {'GW': gw,
                      'gw_entered': True,
                      'gw_points': self.gw_df.loc[self.gw_df['GW'] == gw, 'points'].values[0],
                      'gw_bench_points': self.gw_df.loc[self.gw_df['GW'] == gw, 'points_on_bench'].values[0]}
            picks_gw = self.picks_df.loc[self.picks_df['GW'] == gw, 'picks'].values[0]
            for pick in picks_gw:
                update[f'pick_{pick.pos}_id'] = pick.player_id
                update[f'pick_{pick.pos}_mult'] = pick.multiplier
                update[f'pick_{pick.pos}_c'] = pick.c
                update[f'pick_{pick.pos}_vc'] = pick.vc
                update[f'pick_{pick.pos}_auto_sub'] = pick.autosub
                # find the row index of the player with that id in the player information data frame
                id_row_index = df_player_information.index[df_player_information['id'] == pick.player_id][0]
                update[f'pick_{pick.pos}_web_name'] = df_player_information.at[id_row_index, 'web_name']
                update[f'pick_{pick.pos}_role'] = df_player_information.at[id_row_index, 'element_type']
                # now iterate through the list of history objects
                # find the appropriate game week and get the game week points of that player
                seen_gws = set()
                for points_h in df_player_information.at[id_row_index, 'history']:
                    # Check if the current game week matches the specified one
                    if points_h['gw'] == gw:
                        # Check if the game week has been seen before
                        if points_h['gw'] in seen_gws:
                            if isinstance(points_h['points'], (int, float)):
                                update[f'pick_{pick.pos}_points'] += points_h['points']
                                update[f'pick_{pick.pos}_dgw'] = True
                            else:
                                raise ValueError("points_h['points'] should be a scalar (int or float).")
                        else:
                            update[f'pick_{pick.pos}_points'] = points_h['points']
                            update[f'pick_{pick.pos}_dgw'] = False
                            # Add the game week to the set of seen game weeks
                            seen_gws.add(points_h['gw'])

            updates.append(update)

        # Convert updates to a DataFrame
        updates_df = pd.DataFrame(updates)

        # Merge updates into the original DataFrame
        self.manager_df = self.manager_df.merge(updates_df, on='GW', how='left', suffixes=('', '_update'))

        # Update the original DataFrame with non-NA values from the updates
        for col in updates_df.columns:
            if col != 'GW':
                update_col = f'{col}_update'
                if update_col in self.manager_df.columns:
                    self.manager_df[col] = self.manager_df[col].fillna(self.manager_df[update_col])
                    # Explicitly infer the dtype to handle future downcasting behavior
                    self.manager_df[col] = self.manager_df[col].infer_objects(copy=False)
                    self.manager_df.drop(columns=[update_col], inplace=True)

        # Optionally, defragment the DataFrame by creating a copy
        self.manager_df['gw_entered'] = self.manager_df['gw_entered'].fillna(False)
        self.manager_df = self.manager_df.copy()
        return self.manager_df


class Pick:

    def __init__(self, player_id, pos, multiplier, c, vc, autosub):
        self.player_id = player_id
        self.pos = pos
        self.multiplier = multiplier
        self.c = c
        self.vc = vc
        self.autosub = autosub


class PointHistory:

    def __init__(self, gw, points):
        self.gw = gw
        self.points = points


# end of classes
bootstrap_req = requests.get(f'{league_base_url}bootstrap-static/').json()
# bootstrap gets all the individual player data
player_df = pd.json_normalize(bootstrap_req['elements'])  # elements are each player, normalise makes a df
# python has this weird thing where it defaults to pointers
# the .copy duplicates the df instead of pointing to the existing df.
# element_type is the player position, 1 =gk, 2=def, 3=mid, 4=fwd
df_player_information = player_df[['id', 'web_name', 'element_type']].copy()
df_player_information['history'] = None  # add a column called history to the df

# --- This section of code takes a league and pulls each manager id from it --- #
# get request for the link to the fpl api,
# the f'' allows you to use variables as part of the address
# anything between {} is treated as a variable instead of text
league_req = requests.get(f'{league_base_url}leagues-classic/{league_id}/standings/').json()
# json_normalise turns the json into a dataframe,
# this particular line looks at the "standings" subset of the json,
# which will be used to collect the manager id's for later use.
m_df = pd.json_normalize(league_req['standings'], 'results')
# from the json subset pull the manager id,
# called "entry" in this case rather than "id" as you would expect
df_manager_inf = m_df[['entry']]
# --- --- #


async def fetch_with_retries(session, url):
    retries = 0
    while retries < max_retries:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Rate limit exceeded for URL {url}. Waiting for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                retries += 1
            else:
                print(f"Failed to fetch data from {url}: {response.status}")
                return None
    print(f"Max retries exceeded for URL {url}.")
    return None


async def fetch_manager_data(session, m_id):
    manager_url = f'{manager_base_url}/{m_id}/'
    history_url = f'{manager_base_url}/{m_id}/history/'

    m_manager_data = await fetch_with_retries(session, manager_url)
    if not m_manager_data:
        print(f"Failed to fetch name data for manager id {m_id} after retries.")
        return None

    m_manager_id = m_manager_data['id']  # get the id
    m_team_name = m_manager_data['name']  # get the team name
    m_entered_events = m_manager_data['entered_events']  # get which game weeks they played in

    h_manager_data = await fetch_with_retries(session, history_url)
    if not h_manager_data:
        print(f"Failed to fetch history data for manager id {m_id} after retries.")
        return None

    p_df = pd.DataFrame({'GW': range(1, 39)})  # create a new data frame with 38 game weeks
    p_df['points'] = np.nan  # add a 'points' column to the df
    p_df['points_on_bench'] = np.nan  # add a 'points_on_bench' column to the df
    p_df['picks'] = np.nan  # add a 'picks column to the df
    for event in h_manager_data['current']:  # iterate through each game week the manager played
        gw = event['event']  # the number of the game week
        points = event['points']  # points that gw
        points_on_bench = event['points_on_bench']  # pob that gw
        p_df.loc[p_df['GW'] == gw, 'points'] = points  # add those points to the correct gw in the gw df
        p_df.loc[p_df['GW'] == gw, 'points_on_bench'] = points_on_bench  # as above but pob

    # create a picks df to contain which players a manger picked in a given gw, make a gw column
    e_picks_df = pd.DataFrame({'GW': range(1, 39)})
    e_picks_df['picks'] = None  # make a picks column which will contain an array of pick objects
    for e_gw in m_entered_events:  # iterate through each entered gw
        e_picks = []  # create an empty picks array
        # url for that gw for that manager
        event_url = f'{manager_base_url}/{m_manager_id}/event/{e_gw}/picks/'
        e_response_data = await fetch_with_retries(session, event_url)
        if not e_response_data:
            print(f"Failed to fetch pick data for manager id {m_id} for game week {e_gw} after retries.")
            continue

        sub_ids = [sub['element_in'] for sub in e_response_data['automatic_subs']]  # empty df for substitutes
        # iterate through substitutions
        # add any auto-subs

        for pick in e_response_data['picks']:  # iterate through each pick
            p_id = pick['element']  # get player id
            position = pick['position']  # get position manager placed them 1- 15
            # note this isn't the position the player plays in, that is called "element_type"
            mux = pick['multiplier']  # multiplier 0,1,2 or 3 based on captaincy and bench
            cap = pick['is_captain']  # boolean value for captain
            vice_cap = pick['is_vice_captain']  # boolean value for vice captain
            # use the "multiplier" to figure out if a VC took the armband
            was_autosub = p_id in sub_ids  # if player id matches a autosub id label them autosub true
            e_picks.append(Pick(p_id, position, mux, cap, vice_cap, was_autosub))  # add a new pick to the list

        # place the matching list of picks in the appropriate gw
        e_picks_df.at[e_picks_df.index[e_picks_df['GW'] == e_gw][0], 'picks'] = e_picks

    # finally create a new manager with all the above collected data
    new_manager = Manager(m_manager_id, m_team_name, m_entered_events, p_df, e_picks_df)
    return new_manager  # kick that man back to the async


async def fetch_player_data(session, p_id, df_player_inf, lock):
    player_url = f'{league_base_url}element-summary/{p_id}/'
    retries = 0

    while retries < max_retries:
        async with session.get(player_url) as p_response:
            if p_response.status == 200:
                p_history = []
                p_player_data = await p_response.json()
                for game in p_player_data['history']:  # iterate through player history
                    p_match_points = game['total_points']  # for each week get points
                    p_gw = game['round']  # game week is called "round" in the api
                    p_history.append({"gw": p_gw, "points": p_match_points})

                async with lock:
                    df_player_inf.at[df_player_inf.index[df_player_inf['id'] == p_id][0], 'history'] = p_history
                return  # Exit the function after successful fetch

            elif p_response.status == 429:
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Rate limit exceeded for player id {p_id}. Waiting for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                retries += 1
            else:
                print(f"Failed to fetch data for player id {p_id}: {p_response.status}")
                return None

    print(f"Max retries exceeded for player id {p_id}.")
    return None


async def main():
    global df_player_information  # Add this line to ensure the global DataFrame is used
    lock = asyncio.Lock()
    player_path = os.path.join("C:\\Users\\owenp\\Documents\\JSONS", "players.json")

    # Check if the JSON file exists
    player_file_exists = os.path.exists(player_path)

    if player_file_exists:
        print(f"Player JSON found, loading from file")
        df_player_information = pd.read_json(player_path, orient='records')
    else:
        print(f"No Player JSON found, fetching from API")

    async with aiohttp.ClientSession() as session:
        manager_tasks = []
        player_tasks = []
        if not manager_file_exists:
            manager_tasks = [
                # iterate through each manager id in the manager df to fetch data async
                fetch_manager_data(session, m_id) for m_id in df_manager_inf['entry']
            ]

        if not player_file_exists:
            player_tasks = [
                # iterate through each player id in the player df to find data async
                fetch_player_data(session, p_id, df_player_information, lock) for p_id in df_player_information['id']
            ]

        all_tasks = manager_tasks + player_tasks
        results = await asyncio.gather(*all_tasks)

        for result in results:
            if isinstance(result, Manager):  # Assuming Manager is the class for manager objects
                manager_list.append(result)

        if not player_file_exists:
            # Saving the DataFrame to a JSON file
            df_player_information.to_json(player_path, orient='records', indent=4)
            print(f"DataFrame saved to {player_path}")


def bench_bad_boy():
    headings = ['Team_Name', 'Bench_Sum']
    bench_sums_df = pd.DataFrame(columns=headings)
    for x in manager_list:
        new_row = {'Team_Name': x.name, 'Bench_Sum': x.bench_sum()}
        bench_sums_df = pd.concat([bench_sums_df, pd.DataFrame([new_row])], ignore_index=True)

    max_i = bench_sums_df['Bench_Sum'].idxmax()
    print(bench_sums_df['Team_Name'][max_i])


# Run the main function to fetch data asynchronously
asyncio.run(main())

if manager_file_exists:
    print(f"Manager JSON found, loading from file")
    league_managers_df = pd.read_json(manager_path, orient='records')
else:
    dfs = []
    for manager in manager_list:
        df = manager.construct_data_frame()
        dfs.append(df)

    league_managers_df = pd.concat(dfs, ignore_index=True)
    league_managers_df.to_json(manager_path, orient='records', indent=4)
    print(f"DataFrame saved to {manager_path}")

pg_show(league_managers_df)


