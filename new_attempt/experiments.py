from new_attempt_random_2 import PPricing
import random
import pandas as pd
import matplotlib.pyplot as plt
def experiment_max_evals():
    random.seed(1298)
    df = pd.read_csv('base_supply.csv')
    area_names = df['Start Community Area Name'].unique().tolist()
    new_revenue_list = []
    default_revenue_list = []
    avg_prices = []
    evals = [10]#, 20]#, 50, 100]
    for eval in evals:
        print(f"START: {eval}\n")
        ppricing = PPricing(max_evals=eval)
        ppricing.init_areas(area_names)
        new_revenue, default_revenue, avg_prices = ppricing.run_algo()
        new_revenue_list.append(new_revenue)
        default_revenue_list.append(default_revenue)
    ratio_list = []
    for (default, p_pricing) in zip(default_revenue_list, new_revenue_list):
        l_ratio_list = []
        for (default_value, p_pricing_value) in zip(default, p_pricing):
            l_ratio_list.append(p_pricing_value / default_value)
        ratio_list.append(l_ratio_list)

    fig1, ax1 = plt.subplots()
    for (l_list, eval) in zip(ratio_list, evals):
        ax1.plot(l_list, label=str(eval) + ' evals')
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Ratio of P-Pricing / Default Pricing")
    ax1.legend(loc='best')
    fig1.savefig("total_new_revenue.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_prices, label="Average price")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Price")
    ax2.legend(loc='best')
    fig2.savefig("avg_price_total_evals.png")



def experiment_alpha_beta():
    random.seed(1298)
    df = pd.read_csv('base_supply.csv')
    area_names = df['Start Community Area Name'].unique().tolist()
    new_revenue_list = []
    default_revenue_list = []
    avg_prices = []
    alphas = [0.5]#, 1.0]#, 1.5]
    betas = [0.1]#, 0.2, 0.5]
    for alpha in alphas:
        for beta in betas:
            print(f"START, ALPHA: {alpha}, BETA: {beta}")
            ppricing = PPricing()
            ppricing.init_areas(area_names, alpha=alpha, beta=beta)
            new_revenue, default_revenue, avg_prices = ppricing.run_algo()
            tuple_new = (alpha, beta, new_revenue)
            new_revenue_list.append(tuple_new)
            tuple_default = (alpha, beta, default_revenue)
            default_revenue_list.append(tuple_default)

    ratio_list = []
    for (default, p_pricing) in zip(default_revenue_list, new_revenue_list):
        l_ratio_list = []
        for (default_value, p_pricing_value) in zip(default[2], p_pricing[2]):
            l_ratio_list.append(p_pricing_value / default_value)
        ratio_list.append((default[0], default[1], l_ratio_list))

    fig1, ax1 = plt.subplots()
    for r_list in ratio_list:
        label = f"Alpha: {r_list[0]}, Beta: {r_list[1]}"
        ax1.plot(r_list[2], label=label)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Ratio of P-Pricing / Default Pricing")
    ax1.legend(loc='best')
    fig1.savefig("revenue_alpha_beta")

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_prices, label="Average price")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Price")
    ax2.legend(loc='best')
    fig2.savefig("avg_price_alpha_beta.png")


def experiment_random_demand():
    random.seed(1298)
    df = pd.read_csv('base_supply.csv')
    area_names = df['Start Community Area Name'].unique().tolist()
    new_revenue_list = []
    default_revenue_list = []
    min_values = [0.85]#, 0.90, 0.95]
    max_values = [1.05]#, 1.1]#, 1.15]
    avg_prices = []
    for min in min_values:
        for max in max_values:
            print(f"START: MIN: {min}, MAX: {max}")
            ppricing = PPricing()
            ppricing.init_areas(area_names, min_req=min, max_req=max)
            new_revenue, default_revenue, avg_prices = ppricing.run_algo()
            tuple_new = (min, max, new_revenue)
            new_revenue_list.append(tuple_new)
            tuple_default = (min, max, default_revenue)
            default_revenue_list.append(tuple_default)

    ratio_list = []
    for (default, p_pricing) in zip(default_revenue_list, new_revenue_list):
        l_ratio_list = []
        for (default_value, p_pricing_value) in zip(default[2], p_pricing[2]):
            l_ratio_list.append(p_pricing_value / default_value)
        ratio_list.append((default[0], default[1], l_ratio_list))

    fig1, ax1 = plt.subplots()
    for r_list in ratio_list:
        label = f"Minimum: {r_list[0]}, Maximum: {r_list[1]}"
        ax1.plot(r_list[2], label=label)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Ratio of P-Pricing / Default Pricing")
    ax1.legend(loc='best')
    fig1.savefig("revenue_min_max.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_prices, label="Average price")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Price")
    ax2.legend(loc='best')
    fig2.savefig("avg_price_min_max.png")

experiment_random_demand()