import sys
import pandas as pd
sys.path.insert(0,"..")
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np


# global variables
seller_data = "./data/test1-PreparedSellerData.csv"
buyer_data = "./data/test1-PreparedBuyerData.csv"


# All keys of beta_b_bid_t_mw_mwsegm as numbers not string representation

def optimize():

    # read data from the csv energy seller file, and the csv energy buyer file
    read_csv_data()

    try:        
        # Create a new model
        gurobi_model = gp.Model("mip1")
        
        # Create variables
        # power dispatch level for every lse j with variable bids for every period
        p_j_k = gurobi_model.addVars(lse_id_list, periods, lb=0, name="p_j_k")

        # power dispatch level for every lse, every power segment, every period
        p_n_j_k_mw_bid = dict()
        for lse_bid in price_n_j_k_mw_bid.keys():
            p_n_j_k_mw_bid[lse_bid] = gurobi_model.addVar(lb=0, vtype = GRB.CONTINUOUS, name="p_n_j_k_mw_bid_" + str(lse_bid))


        # contract clearing indicator for every generator g, every period
        c_g_k = gurobi_model.addVars(seller_generator_id_list, periods, vtype=GRB.BINARY, name="c_g_k")

        # power dispatch level for each generator for each period
        p_g_k = gurobi_model.addVars(seller_generator_id_list, periods, lb=0, name="p_g_k")

        # power dispatch level for every generator, every mw_segment, every period
        p_s_g_k_mw_mwseg = dict()
        for offer in phi_s_g_k_mw_mwseg:
            p_s_g_k_mw_mwseg[offer] = gurobi_model.addVar(lb=0, vtype = GRB.CONTINUOUS, name="p_s_g_k_mw_mwseg_" + str(offer))

        # start-up indicator for every generator, every period
        u_g_k = gurobi_model.addVars(seller_generator_id_list, periods, vtype=GRB.BINARY, name="u_g_k")
        
        # set objective function
        gurobi_model.setObjective(
            # benefit from load_serving_entities
            gp.quicksum(
                # price in $/MWh
                price_n_j_k_mw_bid[(n, j, k, mw, bid_id)] * 
                # choosen power in MW
                p_n_j_k_mw_bid[(n, j, k, mw, bid_id)]
            for (n, j, k, mw, bid_id) in price_n_j_k_mw_bid)
            -
            # cost for generators, offer prices (if comitted):
            gp.quicksum(
                # start up costs
                alpha_g_t[(g, k)][0] *
                u_g_k[(g, k)] +
                # no-load costs
                alpha_g_t[(g, k)][1] * 
                c_g_k[(g, k)] for (s, g, k, mw, mwseg) in phi_s_g_k_mw_mwseg
            ) 
            - 
            # cost for generators, performance payments (if comitted):
            gp.quicksum(
                # price in $/MWh
                phi_s_g_k_mw_mwseg[(s, g, k, mw, mwseg)] *
                # chosen power in MW
                p_s_g_k_mw_mwseg[(s, g, k, mw, mwseg)]
                for (s, g, k, mw, mwseg) in phi_s_g_k_mw_mwseg 
            ),
            GRB.MAXIMIZE         
        ) 


        # add constraints

        # constraint 1: power balance constraint (7.14) and on page 95, formula 9.5 and 9.6
        # power production = fixed power consumption + power consumption from bids 
                                # power production 
        gurobi_model.addConstrs(((gp.quicksum(p_g_k[(g, k)] for g in seller_generator_id_list) == 
                            # fixed power consumption
                            (gp.quicksum(lse_fixed_consumption_in_k[(s, timestep)] for (s, timestep) in lse_fixed_consumption_in_k if timestep == k) + 
                            # price-sensitive power consumption from bids
                            gp.quicksum( p_n_j_k_mw_bid[(n, j, timestamp, mw, bid_id)]   for (n, j, timestamp, mw, bid_id) in price_n_j_k_mw_bid if timestamp == k)))
                    for k in periods), "c1"
            )
        

        # find the minimum run power for every generator (asset)
        minimum_run_power = dict()
        for offer in seller_data.iterrows():
            if(minimum_run_power.get( (int(offer[1]["Masked Asset ID"]), int(offer[1]["Trading Interval"])) ) == None):
                minimum_run_power[(int(offer[1]["Masked Asset ID"]), int(offer[1]["Trading Interval"]))] = 0
            minimum_run_power[(int(offer[1]["Masked Asset ID"]), int(offer[1]["Trading Interval"]))] += float(offer[1]["Must Take Energy"])


        # constraint 2.1 power capacity constraint 7.17
        # power production of every generator in a power segment must be smaller than the maximum power capacity of the generator in the power segment
        gurobi_model.addConstrs((p_s_g_k_mw_mwseg[(s, g, k, mw, mwseg)] <= mw * c_g_k[g, k]) for (s, g, k, mw, mwseg) in phi_s_g_k_mw_mwseg)

        # Minimum run load
        gurobi_model.addConstrs( p_g_k[g, k] >= minimum_run_power[(g, k)] * c_g_k[g, k] for (g, k) in minimum_run_power)


        # constraint needed, that the power of a generator is the sum of all power segments of this generator
        gurobi_model.addConstrs( ( p_g_k[generator, timestep] == 
                                gp.quicksum(p_s_g_k_mw_mwseg[(s, generator, timestep, mw, mwseg)] for (s, g, k, mw, mwseg) in phi_s_g_k_mw_mwseg if generator == g and timestep == k)) 
                            for generator in seller_generator_id_list for timestep in periods)
                        

        # constraint needed: each power_consumption of a buyer is smaller or equal to the possible maximum power_consumption of the buyer in the specific period and mw segment
        gurobi_model.addConstrs( p_n_j_k_mw_bid[(n, j, timestamp, mw, bid_id)] <= mw for (n, j, timestamp, mw, bid_id) in price_n_j_k_mw_bid)


        # constraint needed: the sum of power consumptions in the power segments is the power consumption of the buyer in the specific period
        gurobi_model.addConstrs(p_j_k[buyer, timestep] == (gp.quicksum(p_n_j_k_mw_bid[(n, j, t, mw, bid)] for (n, j, t, mw, bid) in price_n_j_k_mw_bid if j == buyer and t == timestep) + 
                                                           float(lse_fixed_consumption_in_k[(buyer, timestep)])) 
                                                    for buyer in lse_id_list for timestep in periods)

        # constraint for buyers needed? 
        # power consumption in each segment only as big as possible     done
        # fixed power consumption always covered                        done
        # 

        # constraint for the start up indicator needed
        # it does't work, that the seller did not start now, is used now but was not used in the period before
        gurobi_model.addConstrs(u_g_k[g, k] - c_g_k[g, k] + c_g_k[g, k-1] >= 0 for g in seller_generator_id_list for k in periods[1:])
        gurobi_model.addConstrs((u_g_k[g, 1] == c_g_k[g, 1]) for g in seller_generator_id_list)
        # constraint, that start up can not be 1 if it was running already
        gurobi_model.addConstrs(u_g_k[g, k] + c_g_k[g, k-1] <= 1 for g in seller_generator_id_list for k in periods[1:])


        # wait for the model to update
        gurobi_model.update()

        # optimize model
        gurobi_model.optimize()


        for v in gurobi_model.getVars():
            print('%s %g' % (v.VarName, v.X))


        # dispose the model and the environment (to create new one in recursive call)
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

        return 
        
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

    except AttributeError:
        print('Encountered an attribute error')
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

    except Exception:
        print('Encountered an exception: ' + str(sys.exc_info()[0]))
        gurobi_model.dispose()
        gp.disposeDefaultEnv()
    
    

def read_csv_data():
 
    read_and_prepare_buyer_data()

    read_and_prepare_seller_data()



def read_and_prepare_buyer_data():
    # READ THE DATA FROM BUYERS FILE (Load Serving Entities LSEs)
    # read data from the csv energy buyer file, and skip the first 4 rows (header)
    global lse_data
    lse_data = pd.read_csv(buyer_data, sep=";", skiprows=4)
    lse_data = lse_data.dropna(axis=1, how='all')

    # remove the first row and the last row (unit of every entry and number of all offers)
    lse_data = lse_data.iloc[1:-1, :]
    lse_data["Masked Lead Participant ID"] = lse_data["Masked Lead Participant ID"].astype(int)
    lse_data["Hour"] = lse_data["Hour"].astype(int)
    # create a list of all the buyers (buyer_ids)
    global lse_id_list
    lse_id_list = lse_data["Masked Lead Participant ID"].unique().tolist()
    lse_id_list = set([int(x) for x in lse_id_list])

    
    # READ THE PERIOD DATA FROM THE BUYER FILE (1 - 24)
    global periods
    periods = sorted(set([int(x) for x in lse_data["Hour"].unique().tolist()]))    
    
    
    # Create the pi_n_j_k_mw_bid  list from buyers, 
    # list of (mw_seg, lse_id, Hour, MW, bid_id) -> price   pairs for each buyer, for each period step (also with multiple MW blocks from one BID)
    # MWssegment is needed, becaus it can be, that a buyer pays for first 200mw 10$/MWh and for the next 200MW 8$/MWh -> Same MW number
    global price_n_j_k_mw_bid
    price_n_j_k_mw_bid = dict()

    # go through the 24 periods
    for k in periods:
        # go through all buyers
        for j in lse_id_list:
            # go through all bids of the buyer in this period (bid_id) -> now i have one row
            for bid in lse_data[(lse_data['Bid Type'] == "PRICE") & (lse_data['Masked Lead Participant ID'] == j) & (lse_data['Hour'] == k)].iterrows():
                # go through every MW step of this bid (there can be 50 MW steps)
                for n in range(1, 51):
                    if(pd.isna(bid[1]["Segment " + str(n) + " MW"])):
                        break

                    price_n_j_k_mw_bid[(n, j, k, float(bid[1]["Segment " + str(n) + " MW"]), int(bid[1]["Bid ID"]))] = float(bid[1]["Segment " + str(n) + " Price"])


    # find all fixed power loads of the buyers in the specific period
    global lse_fixed_consumption_in_k
    lse_fixed_consumption_in_k = dict()
    for lse_bid in lse_data.iterrows():
        #1 fixed power
        # if not in dict jet
        if(lse_fixed_consumption_in_k.get( (int(lse_bid[1]["Masked Lead Participant ID"]), int(lse_bid[1]["Hour"])) ) == None):
            lse_fixed_consumption_in_k[( int(lse_bid[1]["Masked Lead Participant ID"]), int(lse_bid[1]["Hour"]) )] = 0
        # only sum up if FIXED
        if(lse_bid[1]["Bid Type"] == "FIXED" and lse_bid[1]["Segment 1 MW"] != None and lse_bid[1]["Segment 1 MW"] != ""):
            lse_fixed_consumption_in_k[(int(lse_bid[1]["Masked Lead Participant ID"]), int(lse_bid[1]["Hour"]))] += float(lse_bid[1]["Segment 1 MW"])
        


def read_and_prepare_seller_data():
   # READ THE DATA FROM SELLERS FILE
    # read data from the csv energy offers file, and skip the first 6 rows (header)
    global seller_data
    seller_data = pd.read_csv(seller_data, sep=";", skiprows=4)

    # remove the first row and the last row (unit of every entry and number of all offers)
    seller_data = seller_data.iloc[1:-1, :]

    seller_data["Masked Lead Participant ID"] = seller_data["Masked Lead Participant ID"].astype(int)
    seller_data["Masked Asset ID"] = seller_data["Masked Asset ID"].astype(int)
    seller_data["Trading Interval"] = seller_data["Trading Interval"].astype(int)

    # create a list of all the sellers (soller_ids)
    global seller_id_list
    seller_id_list = seller_data["Masked Lead Participant ID"].unique().tolist()
    seller_id_list = sorted(set([int(x) for x in seller_id_list]))

    global seller_generator_id_list
    seller_generator_id_list = seller_data["Masked Asset ID"].unique().tolist()
    seller_generator_id_list = sorted(set([int(x) for x in seller_generator_id_list]))




    # create the offer prices [start up costs, no load costs]
    global alpha_g_t
    alpha_g_t = dict()
    # go through all assets
    for generator in seller_generator_id_list:
        # go through all rows of the seller data for this asset
        for offer in seller_data[(seller_data["Masked Asset ID"] == generator)].iterrows():
            k = int(offer[1]["Trading Interval"])
            if(alpha_g_t.get((generator, k)) == None):
                alpha_g_t[(generator, k)] = [0, 0]
            alpha_g_t[(generator, k)][0] = float(offer[1]["Cold Startup Price"])
            alpha_g_t[(generator, k)][1] = float(offer[1]["No Load Price"])


    # create the performance payment method (seller, generator, period, mw, mwseg) -> price    pairs
    global phi_s_g_k_mw_mwseg
    phi_s_g_k_mw_mwseg = dict()

    
    # go through the 24 periods
    for k in periods:
        # go through all sellers
        for s in seller_id_list:
            # go through all offers of the seller in this period (with ECONMIC BID TYPE (Unit status)) 
            # for offer in seller_data[(seller_data["Masked Lead Participant ID"] == s) & (seller_data["Trading Interval"] == t) & (seller_data["Unit Status"] == "ECONOMIC")].iterrows():
            for offer in seller_data[(seller_data["Masked Lead Participant ID"] == s) & (seller_data["Trading Interval"] == k)].iterrows():
                # go through every MW step of this offer (there can be 10 MW steps)
                for mw_segment in range(1, 11):
                    if(pd.isna(offer[1]["Segment " + str(mw_segment) + " MW"])):
                        break
                    phi_s_g_k_mw_mwseg[(s, int(offer[1]["Masked Asset ID"]), k, float(offer[1]["Segment " + str(mw_segment) + " MW"]), mw_segment)] = offer[1]["Segment " + str(mw_segment) + " Price"]

# Start the main app
if __name__ == "__main__":
    optimize()
    
    '''
    # creating plotting data
    xaxis =[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    yaxis =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # plotting 
    plt.plot(xaxis, yaxis)
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # saving the file.Make sure you 
    # use savefig() before show()
    print("Saving file")
    plt.savefig("./graphs/squares.png")
    '''
