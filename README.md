# TODO



# QUERY BY SQL
python query_by_sql.py output/hl.csv output/cm.csv 'SELECT sedol, citicode, full_description, annual_charge, Wealth150, fidelity_sharpeRatios_oneYear, fidelity_sharpeRatios_threeYear FROM hl_csv
WHERE annual_charge < 0.5 ORDER BY fidelity_sharpeRatios_oneYear DESC LIMIT 5'

# order by fidelity_sharpeRatios_oneYear DESC
python query_by_sql.py output/hl.csv output/cm.csv 'SELECT sedol, citicode, full_description, annual_charge, Wealth150, distribution_yield, underlying_yield, gross_yield, gross_running_yield, fidelity_sharpeRatios_oneYear, fidelity_sharpeRatios_threeYear FROM hl_csv
WHERE annual_charge < 0.5 and Wealth150=1 ORDER BY fidelity_sharpeRatios_oneYear DESC LIMIT 15'

# order by distribution_yield DESC
python query_by_sql.py output/hl.csv output/cm.csv 'SELECT sedol, citicode, full_description, annual_charge, Wealth150, distribution_yield, underlying_yield, gross_yield, gross_running_yield, fidelity_sharpeRatios_oneYear, fidelity_sharpeRatios_threeYear FROM hl_csv WHERE annual_charge < 0.5 and Wealth150=1 ORDER BY distribution_yield DESC LIMIT 15'

# order by running_yield DESC
python query_by_sql.py output/hl.csv output/cm.csv 'SELECT sedol, citicode, full_description, annual_charge, Wealth150, distribution_yield, underlying_yield, gross_yield, gross_running_yield, fidelity_sharpeRatios_oneYear, fidelity_sharpeRatios_threeYear FROM hl_csv WHERE annual_charge < 0.5 ORDER BY running_yield DESC LIMIT 15'

#
Running yield and distribution yield differ primarily in timing and payment structure:

Running Yield: Reflects current market price movements, updating daily.
Distribution Yield: Represents actual cash payments made to investors over a specific period.