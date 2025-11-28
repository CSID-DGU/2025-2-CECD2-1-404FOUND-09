import { Inter } from "next/font/google";
import GlobalStyles from "@/styles/GlobalStyles";
import FedNavBar from "@/components/navigator/FedNavBar";
import { Toaster } from "sonner";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "FedHB - 연합학습 하이브리드 데모 플랫폼",
  description: "차세대 연합 학습 알고리즘, FedHB입니다",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <GlobalStyles />
        <FedNavBar />
        <main style={{ paddingTop: "60px" }}>{children}</main>
        <Toaster />
      </body>
    </html>
  );
}
