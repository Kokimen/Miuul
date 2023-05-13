import streamlit as st
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

st.set_page_config(page_title = "Motivational Letter for Data and Educational Company Miuul", page_icon = ":tada", layout = "wide")

with st.expander("1 - Ben Kimim?"):
    st.subheader("Burak Koktay 👀")
    st.title("Jr. Data Analyst and Digital Marketing Researcher")
    st.write("Hem Product Marketing Manager hem de aylardır bilgiler öğrendiğim insanın açtığı pozisyona başvurmaktan heyecan duyuyorum. Geçmişimdeki pazarlama deneyimim ve "
             "miuul ailesinden olduğum veri eğitimleri sayesinde, ekibinize değerli katkılar sağlayabileceğime eminim.")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_mlp3zxve.json")
lottie_codingv2 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_xgemoqmm.json")

with st.expander("2 - Neden Başvuruyorum?"):
    with st.container():
        # st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Neden başvuruyorum?")
            st.write(
                """
                - On-Site olmayan işlerde daha mutlu, bu yüzden daha da verimli bir çalışan olduğum için
                - Şuanki işimde kendime katacağım bir şey kalmadığı için, daha yeni ve engellerle dolu bir işte çalışarak kendime değer katmak
                - Teknolojiden sonuna kadar faydalanmanın yanında veriye dayalı hedefleri olan bir şirkette görev almak
                - Yurt dışı ayağı olan bir şirkette çalışmak istediğim için
                """
            )
        with right_column:
            st_lottie(lottie_coding, height = 300, key = "coding")

with st.expander("3 - Seni Neden İşe Alalım?"):
    with st.container():
        # st.write("---")
        left_column, right_column = st.columns(2)
        with right_column:
            st.header("Seni neden işe alalım?")
            st.write(
                """
                - Miuul kökenli olduğum için ürün/hizmetlere, programlara ve içeriklere hakimiyet
                - Şuan çalıştığım şirketin yalnızca Avrupa'ya hizmet sağlamasından dolayı, İngilizce yazıyor ve konuşuyor olmam
                - Avrupa vatandaşı ve pasaportuna sahip olduğum için her an yurt dışı operasyonlarında bulunabilme 
                - Akademide pazarlama alanında 3 yıl tecrübeye sahip olduğum için akademik bilgilerimi kullanarak şirkete yeni bakış açıları getirebilme
                - Üniversite döneminde Eczacıbaşı ve Petkim gibi büyük şirketler için halkla ilişkiler, pazarlama ve reklamcılık alanlarında projeler yürütmüş olmam
                - Bu projeler sırasında müşteri segmentasyonu, pazar analizi, rakip analizi ve SWOT analizi gibi yöntemleri kullanmış biri olmam
                - Yüksek lisans tezimin ve makalelerimin genellikle tüketici davranışları ve karar mekanizmaları üzerine olması
                - Veri bilimi, yapay zeka, chatGPT gibi konularda aylarca süren Coursera eğitimlerine ek olarak, Vahit Keskin'den veri eğitimi almış olmam
                """
            )
        with left_column:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st_lottie(lottie_codingv2, height = 300, key = "codingv2")

with st.expander("4 - Pazarlamacı mısın, Veri Bilimci misin Anlamadık?"):
    st.header("Pazarlamacı mısın, veri bilimci misin anlamadık?")
    st.write("Güncel olarak dış ticaret departmanında iş analisti olarak çalışırken, hem dijital pazarlama alanında doktora yapıyorum hem de veri bilimi hakkında "
             "bilgilerimi geliştirmeye çalışıyorum.Şirketlerin itibarlarını "
             "geliştirme, ürün/hizmetlerini yaygınlaştırma amacı ile ajanslarda çalışmak istiyordum. Ancak bu sektör için gelen bütün iş tekliflerinin "
             "İstanbul özelinde olmasında ve İstanbul'da hayatımı sürdürmek istemememden dolayı ajanslarda çalışma fırsatı bulamadım ve akademik yanım kuvvetli olduğu için akademiyi seçtim."
             "Akademik network'ün desteği sayesinde TÜBİTAK'da veri analizcisi olarak bir süre görev aldım, projelere destek verdim. Bu esnada verilerle çalışmaktan keyif aldığımı "
             "fark ettim. Dolayısıyla pazarlama ve reklamcılık alanında veriye dayalı karar verme mekanizmalarına başvurulan departmanlarda aktif rol alabilecek yetkinliklere "
             "sahibim. Şirket dinamikleri (sorumluluk, itibar, imaj vb.) ve pazarlama sektör bilgisine sahip bir vericiyim diyelim.")

with st.expander("5 - Pozisyona Uygun Akademik Araştırmalarım"):
    st.header("Pozisyona uygun akademik araştırmalarım")
    st.write(
        """
        - The Impact of Personal and Contextual Features of Social Media Phenomena on Consumers: Influencer Marketing
        - The Effect of Using a Same Celebrity in Advertising of Various Brands in the Same Period on Consumer Attitudes
        - The Impact of Personal Factors on Consumer Decision Styles: A Comparison of Y and Z GenerationThe Impact of Personal Factors on Consumer Decision Styles: A Comparison of Y and Z Generation
        """
    )
# yillar = [2017, 2018, 2019, 2020]
# deneyim = [0, 1, 2, 3]
#
# plt.figure(figsize = (2, 2))
# plt.plot(yillar, deneyim, "-o")
# plt.xlabel('Yıllar')
# plt.ylabel('Deneyim Yılı')
# plt.title("Akdeniz Üniversitesi Dijital Pazarlama ve Reklamcılık Araştırma Asistanlığı")
# plt.xticks(yillar)
#
# st.pyplot(plt.gcf())
