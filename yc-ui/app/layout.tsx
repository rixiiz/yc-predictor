import "./globals.css";

export const metadata = {
  title: "YC Predictor",
  description: "Predict YC-likeness from a YouTube pitch video (probability only).",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <div className="container">{children}</div>
      </body>
    </html>
  );
}
