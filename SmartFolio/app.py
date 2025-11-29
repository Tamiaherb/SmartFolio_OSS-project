import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf # [추가] 고급 통계 기법

# --- 1. 기본 설정 ---
st.set_page_config(page_title="SmartFolio", page_icon="📈", layout="wide")

st.title("📈 SmartFolio: 포트폴리오 최적화 배분기")
st.markdown("""
**현대 포트폴리오 이론(MPT: Modern Portfolio Theory)**을 기반으로 당신의 투자 성향에 맞는 최적의 자산 배분 비율을 제안합니다.
주식 종목을 입력하고, 수학적으로 증명된 **'황금 비율'**을 찾아보세요.
""")

# --- 2. 사이드바 (사용자 입력) ---
st.sidebar.header("🔧 설정")
tickers_input = st.sidebar.text_input("종목 티커 (쉼표로 구분)", "005930.KS, 000660.KS, 035420.KS, 035720.KS")
st.sidebar.caption("예: 삼성전자(005930.KS), Apple(AAPL)")

start_date = st.sidebar.date_input("분석 시작일", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("분석 종료일", pd.to_datetime("2024-01-01"))

if st.sidebar.button("🚀 분석 시작"):
    tickers = [t.strip() for t in tickers_input.split(',')]
    
    if len(tickers) < 2:
        st.error("최소 2개 이상의 종목을 입력해야 포트폴리오를 구성할 수 있습니다.")
    else:
        try:
            with st.spinner('데이터를 수집하고 최적의 비율을 계산 중입니다...'):
                # --- 3. 데이터 다운로드 ---
                data = yf.download(tickers, start=start_date, end=end_date)['Close']
                if data.empty:
                    st.error("데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
                    st.stop()

                # 수익률 및 통계 계산
                daily_returns = data.pct_change().dropna() #일간 변동률계싼
                mean_returns = daily_returns.mean() * 252  # 연간 기대 수익률
               # (수정)Shrinkage Covariance (Ledoit-Wolf) 적용
                # 일반적인 sample_cov보다 노이즈에 robust
                lw = LedoitWolf()
                # sklearn은 (n_samples, n_features)를 원함
                lw.fit(daily_returns) 
                cov_matrix = lw.covariance_ * 252 
                # 다시 DataFrame으로 변환 (인덱스 유지를 위해)
                cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

                # --- 4. 포트폴리오 최적화 (MPT 핵심 로직) ---
                def portfolio_performance(weights, mean_returns, cov_matrix):
                    returns = np.sum(mean_returns * weights)
                    # 행렬 연산 시 DataFrame 대신 numpy array 사용 권장
                    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                    return returns, std

                # 샤프 지수(수익/위험)를 최대화하는 목적 함수 (음수로 변환하여 최소화 문제로 풂)
                def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
                    p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
                    return -(p_ret / p_var)

                # 제약 조건: 가중치의 합은 1, 각 가중치는 0~1 사이
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(len(tickers)))
                init_guess = [1./len(tickers) for _ in range(len(tickers))]

                # Scipy 최적화 실행
                opt_result = sco.minimize(neg_sharpe_ratio, init_guess, 
                                        args=(mean_returns, cov_matrix), 
                                        method='SLSQP', bounds=bounds, constraints=constraints)
                
                best_weights = opt_result.x
                best_ret, best_vol = portfolio_performance(best_weights, mean_returns, cov_matrix)
                best_sharpe = best_ret / best_vol

                # --- 5. 결과 시각화 ---
                st.success("분석 완료! 최적의 포트폴리오를 발견했습니다.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("기대 연수익률", f"{best_ret*100:.2f}%")
                col2.metric("예상 리스크 (변동성)", f"{best_vol*100:.2f}%")
                col3.metric("샤프 지수 (효율성)", f"{best_sharpe:.2f}")

                tab1, tab2, tab3, tab4 = st.tabs(["📊 최적 배분", "📈 주가 추이", "🔥 리스크 분석", "💰 백테스팅"])

                with tab1:
                    st.subheader("제안하는 자산 배분 비율(Ledoit-Wolf 수축 추정량을 사용하여 outlier에 더 강건한 비중을 산출")
                    # 파이 차트
                    df_weights = pd.DataFrame({'종목': tickers, '비중': best_weights})
                    fig_pie = px.pie(df_weights, values='비중', names='종목', hole=0.4)
                    fig_pie.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with tab2:
                    st.subheader("지난 기간 주가 변동")
                    # 정규화된 그래프 (100에서 시작)
                    norm_data = data / data.iloc[0] * 100
                    st.line_chart(norm_data)

                with tab3:
                    st.subheader("종목 간 상관관계 (Correlation)")
                    st.write("색이 진한 빨간색일수록 두 종목이 비슷하게 움직입니다. (분산투자 효과 낮음)")
                    corr_matrix = data.pct_change().corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                with tab4:
                    st.subheader("💰 백테스팅: 과거 수익률 시뮬레이션")
                    st.markdown("**'만약 이 비율대로 1,000만 원을 투자했다면?'**")
                    
                    initial_investment = 10000000 # 1,000만원 가정
                    
                    # 1. 내 포트폴리오 가치 변화 계산
                    # 정규화된 데이터(1.0 시작)에 초기자금 곱하기
                    # 각 종목별 보유 금액 = (초기자금 * 비중) * (가격변동배율)
                    # 전체 자산 = 종목별 보유 금액의 합
                    
                    # (날짜, 종목) * (종목 비중) -> (날짜, 종목별 가치)
                    price_change = data / data.iloc[0] # 1.0부터 시작하는 배율
                    # (가격배율)*(프로그램이 정해준 비중)*(내 원금) = 종목별 현재 평가금
                    #.sum(axis=1) = 종목별 평가금을 다 더해서 '내 총자산' 계산
                    portfolio_value = (price_change * best_weights * initial_investment).sum(axis=1)
                    
                    # 2. 벤치마크 (1/N 균등 투자) 가치 변화 계산
                    equal_weights = np.array([1/len(tickers)] * len(tickers))
                    benchmark_value = (price_change * equal_weights * initial_investment).sum(axis=1)
                    
                    # 3. 데이터프레임 합치기
                    backtest_df = pd.DataFrame({
                        'AI 최적화 포트폴리오': portfolio_value,
                        '단순 균등 투자 (1/N)': benchmark_value
                    })
                    
                    # 4. 시각화
                    st.line_chart(backtest_df)
                    
                    # 5. 최종 결과 요약
                    final_ai = portfolio_value.iloc[-1]
                    final_bm = benchmark_value.iloc[-1]
                    
                    col_b1, col_b2 = st.columns(2)
                    col_b1.metric("AI 포트폴리오 최종 금액", f"{int(final_ai):,}원", 
                                  delta=f"{((final_ai/initial_investment)-1)*100:.1f}%")
                    col_b2.metric("단순 투자 최종 금액", f"{int(final_bm):,}원",
                                  delta=f"{((final_bm/initial_investment)-1)*100:.1f}%")

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.warning("티커가 정확한지 확인해주세요. 한국 주식은 끝에 .KS를 붙여야 합니다.")
else:
    st.info("👈 왼쪽 사이드바에서 종목을 입력하고 분석 버튼을 눌러주세요.")