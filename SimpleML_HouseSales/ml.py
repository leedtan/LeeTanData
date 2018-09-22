def plot_by_group(yhat_all, val_Y_all, increment=50, metric='abs'):

    if metric == 'abs':
        err = np.abs((yhat_all - val_Y_all) / val_Y_all)
        title = "Absolute Percent Errors by house sales group"
    else:
        err = (yhat_all - val_Y_all) / val_Y_all
        title = "Raw Percent Errors by house sales group"

    true_bin = ((val_Y_all // increment) + 1) * increment
    err_bins = {}
    for pe, tb in zip(err, true_bin):
        if not tb in err_bins.keys():
            err_bins[tb] = []
        err_bins[tb] += [pe * 100]

    err_list = [err_bins[key] for key in sorted(err_bins)]
    medians = [np.median(err_list[idx]) for idx in range(len(err_list))]
    q25s = [np.percentile(err_list[idx], 25) for idx in range(len(err_list))]
    q75s = [np.percentile(err_list[idx], 75) for idx in range(len(err_list))]
    keys = [key for key in sorted(err_bins)]
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15, 5))

    ax.set(
        xlabel='true time to sell house',
        ylabel='IQR of percent error by group',
        title=title)

    # Add std deviation bars to the previous plot
    ax.errorbar(
        keys,
        medians,
        yerr=[[medians[i] - q25s[i] for i in range(len(medians))],
              [q75s[i] - medians[i] for i in range(len(medians))]],
        fmt='-o')
    xticks = [
        str(key - 50)[:-2] + ' to ' + str(key)[:-2] + '\n' + 'size group:' +
        str(len(err_list[idx])) for idx, key in enumerate(sorted(err_bins))
    ]
    ax.set_xlim([keys[0] - increment / 2, keys[-1] + increment / 2])
    ax.set_xticks(np.arange(np.min(keys), np.max(keys) + 1, increment))
    ax.set_xticklabels(xticks)
    plt.show()
    return err_bins
