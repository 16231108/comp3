import kfp
from kfp import dsl
from kfp.dsl import ContainerOp
from kfp.dsl import InputArgumentPath
def fetch_data():
    return ContainerOp(
        name='fetch-data',
        image='star16231108/baseline:1.1',
        command=['python'],
        arguments=["main.py",'fetch'],
        file_outputs={'out': '/result'},
        )
def feature_engineering(data):
    return ContainerOp(
        name='feature-engineering',
        image='star16231108/baseline:1.1',
        command=['python'],
        arguments=['main.py','feature',InputArgumentPath(data)],
        file_outputs={'train_out':'/result_train',
        'trade_out':'/result_trade'}
        )
def train_model(train_df,trade_df):
    return ContainerOp(
        name='train-model',
        image='star16231108/baseline:1.1',
        command=['python'],
        arguments=['main.py','train_model',
        InputArgumentPath(train_df),
        InputArgumentPath(trade_df)],
        file_outputs={'trained_model':'/model.pkl',
        'e_trade_gym':'/e_trade.pkl'})
def tradeing(m_path,t_path):
    return ContainerOp(
        name='tradeing',
        image='star16231108/baseline:1.1',
        command=['python'],
        arguments=['main.py','trade',
        InputArgumentPath(m_path),
        InputArgumentPath(t_path)],
        file_outputs={'df_account_value':'/df_account_value'}
        )
def result_backtest(a_value):
    return ContainerOp(
        name='Get Backtest all Results',
        image='star16231108/baseline:1.1',
        command=['python'],
        arguments=['main.py','backtest',
        InputArgumentPath(a_value)],
        file_outputs={'perf_stats_all','/perf_stats_all'}
        )
@dsl.pipeline(
    name='FinRL-Library-2',
)
def sequential_pipeline():
    data=fetch_data()
    feature=feature_engineering(data.outputs['out'])
    model=train_model(feature.outputs['train_out'],
        feature.outputs['trade_out'])
    trade=tradeing(model.outputs['train_model'],model.outputs['e_trade_gym'])
    backtest=result_backtest(trade.outputs['df_account_value'])
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')
